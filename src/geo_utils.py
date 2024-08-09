import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import osmnx as ox
import re
import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import warnings
from scipy.spatial.distance import cdist
import itertools
from typing import Any, List, Dict, Optional, Union, Tuple
from vistdf.images import ImagesFromAPI

PARCEL_URL = "https://data.cityofberkeley.info/api/geospatial/bhxd-e6up?method=export&format=GeoJSON"

def heading(line: LineString) -> float:
    """
    Calculate the heading (direction) of a LineString in degrees.

    Args:
    - line (LineString): The input LineString.

    Returns:
    - float: The heading of the LineString in degrees.
    """
    start_point, end_point = line.coords[0], line.coords[-1]
    delta_lon, delta_lat = abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1])
    angle = math.atan2(delta_lon, delta_lat)
    return (math.degrees(angle) + 360) % 360

def perpendicular_headings(line: LineString, original_heading: float, round_base: int = 5) -> Tuple[float, float]:
    """
    Calculate the perpendicular headings to the original heading.

    Args:
    - line (LineString): The input LineString.
    - original_heading (float): The original heading in degrees.
    - round_base (int): The base to which the headings should be rounded. Default is 5.

    Returns:
    - Tuple[float, float]: The two perpendicular headings.
    """
    heading1 = round5((original_heading + 90) % 360, base=round_base)
    heading2 = round5((original_heading - 90 + 360) % 360, base=round_base)
    return min(heading1, heading2), max(heading1, heading2)

def interpolate(line: LineString, distance: float) -> List[Point]:
    """
    Interpolate points along a LineString at a specified distance.

    Args:
    - line (LineString): The input LineString.
    - distance (float): The distance between interpolated points.

    Returns:
    - List[Point]: List of interpolated points.
    """
    num_points = int(line.length / distance) + 1
    distances = np.linspace(0, line.length, num_points)
    points = [line.interpolate(distance) for distance in distances]
    return points

def round5(x: float, base: int = 5) -> float:
    """
    Round a number to the nearest multiple of the base.

    Args:
    - x (float): The input number.
    - base (int): The base to which the number should be rounded.

    Returns:
    - float: The rounded number.
    """
    return base * round(x/base)

def load_parcels(parcels_url: str = PARCEL_URL) -> gpd.GeoDataFrame:
    """
    Load parcel data from a specified URL.

    Args:
    - parcels_url (str): The URL to the parcel data.

    Returns:
    - gpd.GeoDataFrame: The loaded parcel data.
    """
    parcels = gpd.read_file(parcels_url)
    parcels['coords'] = gpd.GeoSeries([Point(xy) for xy in zip(parcels['longitude'], parcels['latitude'])])
    parcels = parcels[['situs_addre', 'parcelid', 'geometry', 'coords']]
    return parcels

def clip_gdf_to_bbox(gdf: gpd.GeoDataFrame, bbox: Polygon, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to a specified bounding box.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.
    - bbox (Polygon): The bounding box to which the GeoDataFrame should be clipped.
    - crs (str): The coordinate reference system.

    Returns:
    - gpd.GeoDataFrame: The clipped GeoDataFrame.
    """
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}).set_crs(crs)
    bounded_gdf = gpd.overlay(gdf, bbox_gdf, how='intersection')
    return bounded_gdf

def generate_network_pts(bbox: Polygon, api_key: str, samp_dist: float = 0.00015, min_road_len: int = 5) -> gpd.GeoDataFrame:
    """
    Generate network points within a bounding box.

    Args:
    - bbox (Polygon): The bounding box.
    - samp_dist (float): The sampling distance between points. Default is 0.00015.
    - min_road_len (int): The minimum length of roads to consider. Default is 5.

    Returns:
    - gpd.GeoDataFrame: The generated network points.
    """
    G = ox.graph.graph_from_polygon(bbox)
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges[edges['highway'].isin(['residential', 'tertiary'])][['osmid', 'length', 'geometry']]
    edges = edges[edges['length']>min_road_len]
    edges['points'] = edges.apply(lambda row: interpolate(row['geometry'], samp_dist), axis=1)
    edges['heading'] = edges['geometry'].apply(lambda x: round(heading(x)))
    pts = edges.explode('points')
    pts[['meta_pt', 'pano_id']] = load_gsv_meta_from_coords(pts, api_key).apply(pd.Series)
    pts['meta_pt_str'] = pts['meta_pt'].astype(str)
    filtered_pts = pts.groupby(['meta_pt_str', 'pano_id']).apply(filter_headings).reset_index(drop=True).drop(['meta_pt_str'], axis=1)
    filtered_pts['perp_heading'] = filtered_pts.apply(lambda row: perpendicular_headings(row['geometry'], row['heading']), axis=1)
    return filtered_pts.explode('perp_heading')

def get_area_bbox(area: str, api_key: str) -> Polygon:
    """
    Get the bounding box for a specified area.

    Args:
    - area (str): The name of the area.
    - key (str): The Google Maps API key.

    Returns:
    - Polygon: The bounding box of the area.
    """
    resp = requests.get(
        url='https://maps.googleapis.com/maps/api/geocode/json',
        params={'address': re.sub(r'\s*,\s*', ',+', area), 'key': api_key}
    )
    resp.raise_for_status()

    data_resp = resp.json()
    if data_resp.get('status') != 'OK':
        raise ValueError(f"API error: {data_resp.get('status')}")

    results = data_resp.get('results', [])
    if not results:
        raise ValueError("No results found")

    bounds = results[0].get('geometry', {}).get('bounds')
    bbox = box(bounds['southwest']['lng'], bounds['southwest']['lat'], bounds['northeast']['lng'], bounds['northeast']['lat'])
    return bbox

def new_pos(point: Point, heading: float, distance: float) -> Point:
    """
    Calculate a new position from a point, heading, and distance.

    Args:
    - point (Point): The starting point.
    - heading (float): The heading in degrees.
    - distance (float): The distance to travel.

    Returns:
    - Point: The new position.
    """
    heading_rad = math.radians(heading)
    delta_x = distance * math.sin(heading_rad)
    delta_y = distance * math.cos(heading_rad)
    return Point(point.x + delta_x, point.y + delta_y)

def filter_headings(group: pd.DataFrame, tolerance: float = 45) -> pd.DataFrame:
    """
    Filter headings to ensure a minimum separation.

    Args:
    - group (pd.DataFrame): The input DataFrame containing headings.
    - tolerance (float): The minimum separation between headings.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    group = group.sort_values(by='heading')
    filtered_group = [group.iloc[0]]
    for _, row in group.iterrows():
        if all(abs(row['heading'] - x['heading']) > tolerance for x in filtered_group):
            filtered_group.append(row)
    return pd.DataFrame(filtered_group)

def separate_points_headings(gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    """
    Separate points and headings into individual rows.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame containing points and headings.
    - distance (float): The distance to use for separation.

    Returns:
    - gpd.GeoDataFrame: The separated GeoDataFrame.
    """
    separated_data = [
        (new_pos(row['points'], heading, distance), heading, row)
        for _, row in gdf.iterrows()
        for heading in (row['heading_1'], row['heading_2'])
    ]

    separated_points, headings, original_rows = zip(*separated_data)
    original_columns = original_rows[0].index

    data_dict = {col: [row[col] for row in original_rows] for col in original_columns}
    data_dict['geometry'] = separated_points
    data_dict['heading'] = headings

    return gpd.GeoDataFrame(data_dict, crs=gdf.crs).drop(['points', 'heading_1', 'heading_2'], axis=1)

def join_parcels(pts: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Join points with parcels using a nearest spatial join.

    Args:
    - pts (gpd.GeoDataFrame): The input points GeoDataFrame.
    - parcels (gpd.GeoDataFrame): The input parcels GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: The joined GeoDataFrame.
    """
    joined_gdf = pts.sjoin_nearest(parcels, how='inner', distance_col='distance')

def load_gsv_meta_from_coords(
    df: pd.DataFrame, api_key: str, pt_label: str = 'points', size: str = '640x640') -> pd.Series:
    """
    Load Google Street View metadata from coordinates.

    Args:
    - df (pd.DataFrame): The input DataFrame containing coordinates.
    - pt_label (str): The column name containing points.
    - size (str): The size of the image.
    - key (str): The Google Maps API key.

    Returns:
    - pd.Series: A series containing metadata for each coordinate.
    """
    def get_single_gsv(row):
        params = {
            'size': size,
            'location': f"{row[pt_label].y},{row[pt_label].x}",
            'key': api_key,
        }

        url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return Point(data['location']['lng'], data['location']['lat']), data['pano_id']

    tqdm.pandas()
    return df.progress_apply(get_single_gsv, axis=1)

def load_gsv_img_from_coords(
    df: pd.DataFrame,
    api_key: str,
    save_dir='gsv_images',
    pt_label: str = 'points',
    heading_label: str = 'perp_heading',
    size: str = '640x640'
) -> "pd.DataFrame":
    """
    Load Google Street View images from coordinates.

    Args:
    - df (pd.DataFrame): The input DataFrame containing coordinates.
    - pt_label (str): The column name containing points.
    - heading_label (str): The column name containing headings.
    - size (str): The size of the image.
    - key (str): The Google Maps API key.

    Returns:
    - None
    """
    os.makedirs(save_dir, exist_ok=True)

    def get_single_gsv(idx):
        pano_id, heading = idx.split("_")

        params = {
            'size': size,
            'pano': pano_id,
            'key': api_key,
            'heading': heading
        }

        url = 'https://maps.googleapis.com/maps/api/streetview'
        response = requests.get(url, params=params)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert('RGB')
        return np.asarray(image)[:, :, ::-1]

    ids = (df.pano_id.map(str) + "_" + df.heading).to_list()
    images = ImagesFromAPI(ids, get_single_gsv, save_dir)

    tqdm.pandas()
    img_series = pd.Series([*images])
    return df.assign(image=img_series)