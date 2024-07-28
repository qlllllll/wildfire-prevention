import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import itertools

import cv2
import torch
from torchvision.ops import box_convert
import torchvision.ops as ops

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    """
    Load an image from a file path or URL.

    Args:
    - image_str (str): File path or URL of the image.

    Returns:
    - Image.Image: Loaded image.
    """
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.

    Args:
    - image (Image.Image): Input image.
    - labels (List[str]): List of labels to detect.
    - threshold (float): Detection threshold.
    - detector_id (Optional[str]): Detector model ID.

    Returns:
    - List[Dict[str, Any]]: Detection results.
    """
    if isinstance(image, str):
        image = load_image(image)
        
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=DEVICE)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    detections = object_detector(image,  candidate_labels=labels, threshold=threshold)
    
    boxes, scores = [], []

    for detection in detections:
        box = detection['box']
        boxes.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
        scores.append(detection['score'])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    iou_threshold = 0.5

    keep_indices = ops.nms(boxes, scores, iou_threshold)

    detections = [detections[i] for i in keep_indices]
    
    results = [DetectionResult.from_dict(result) for result in detections]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image and a set of bounding boxes.

    Args:
    - image (Image.Image): Input image.
    - detection_results (List[Dict[str, Any]]): Detection results.
    - polygon_refinement (bool): Whether to refine the masks using polygons.
    - segmenter_id (Optional[str]): Segmenter model ID.

    Returns:
    - List[DetectionResult]: Detection results with masks.
    """
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(DEVICE)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(DEVICE)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = True,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    """
    Perform grounded segmentation on an image.

    Args:
    - image (Union[Image.Image, str]): Input image.
    - labels (List[str]): List of labels to detect.
    - threshold (float): Detection threshold.
    - polygon_refinement (bool): Whether to refine the masks using polygons.
    - detector_id (Optional[str]): Detector model ID.
    - segmenter_id (Optional[str]): Segmenter model ID.

    Returns:
    - Tuple[np.ndarray, List[DetectionResult]]: Segmentation results.
    """
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return detections

def obj_detection(image_series: pd.Series, text_prompt: List[str]) -> pd.Series:
    """
    Perform object detection on a series of images with a text prompt.

    Args:
    - image_series (pd.Series): Series of images.
    - text_prompt (str): Text prompt for object detection.

    Returns:
    - pd.Series: Series of detection results.
    """
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    
    detections = image_series.apply(lambda img: grounded_segmentation(image=img, labels=text_prompt, threshold=0.3, polygon_refinement=True, detector_id=detector_id,
    segmenter_id=segmenter_id))

    return detections 

def reformat_detections(detections: pd.Series) -> Dict[str, pd.Series]:
    """
    Reformat detections into a structured DataFrame.

    Args:
    - detections (pd.Series): Series of detection results, where each element is a list of DetectionResult objects.

    Returns:
    - Dict[str, pd.Series]: Dictionary of pd.Series grouped by label, with each pd.Series containing masks and the original index.
    """
    obj = detections.reset_index(name='detections').explode('detections')
    
    data = [{
        'label': row['detections'].label,
        'mask': row['detections'].mask,
        'original_index': row['index']
    } for _, row in obj.iterrows()]
    
    full_df = pd.DataFrame(data)
    grouped = full_df.groupby('label')
    
    return {label: group.drop(['label'], axis=1).groupby('original_index').agg(list)['mask'] for label, group in grouped}

def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 90 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 90 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def depth_to_points(depth, R=None, t=None):
    depth = np.array([depth])
    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0].reshape(-1, 3)

def depth_estimate(img_series: pd.Series) -> pd.Series:
    """
    Estimate depth from a series of images.

    Args:
    - img_series (pd.Series): Series of images.

    Returns:
    - pd.Series: Series of depth maps.
    """
    repo = "isl-org/ZoeDepth"
    model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    zoe = model_zoe_k.to(DEVICE)
    return img_series.apply(zoe.infer_pil)

def estimate_3d(depth_series: pd.Series, est_func=depth_to_points) -> pd.Series:
    """
    Estimate 3D points from depth maps.

    Args:
    - depth_series (pd.Series): Series of depth maps.
    - est_func (callable): Function to convert depth maps to 3D points.

    Returns:
    - pd.Series: Series of 3D points.
    """
    return depth_series.apply(est_func)

def bound_3d(row: pd.Series) -> Optional[List[List[Tuple[float, float, float]]]]:
    """
    Generate 3D bounding boxes from masks and coordinates.

    Args:
    - masks (List[np.ndarray]): List of masks.
    - coords (np.ndarray): Coordinates.

    Returns:
    - Optional[List[List[Tuple[float, float, float]]]]: List of bounding boxes.
    """
    masks, coords = row[0], row[1]
    if masks is None:
        return None 
    
    corners = [
        generate_bounding_box_corners(
            np.concatenate([
                np.min(image_3d_coords := coords[mask.flatten() == 255], axis=0),
                np.max(image_3d_coords, axis=0)
            ])
        )
        for mask in masks
    ]

    return corners

def estimate_obj_bounds(mask_series: pd.Series, coords_series: pd.Series, label: str) -> pd.Series:
    """
    Generate 3D bounding boxes for each object.

    Args:
    - mask_series (pd.Series): Series containing the masks.
    - coords_series (pd.Series): Series containing the coordinates.
    - label (str): The label for the bounding boxes.

    Returns:
    - pd.Series: Series containing the 3D bounding boxes, with the name set to the label.
    """
    merged = mask_series.merge(coords_series, left_index=True, right_index=True)
    bounds = merged.apply(bound_3d, axis=1)
    bounds.name = label

    return bounds

def generate_bounding_box_corners(bounds: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Generate corners of a bounding box.

    Args:
    - bounds (np.ndarray): Bounding box coordinates.

    Returns:
    - List[Tuple[float, float, float]]: List of corners.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return list(itertools.product([min_x, max_x], [min_y, max_y], [min_z, max_z]))

def closest_distances(row: pd.Series) -> Optional[List[float]]:
    """
    Calculate closest distances between two sets of bounding boxes.

    Args:
    - row (pd.Series): Series containing two elements:
      - bounds1 (List[List[Tuple[float, float, float]]]): First set of bounding boxes.
      - bounds2 (List[List[Tuple[float, float, float]]]): Second set of bounding boxes.

    Returns:
    - Optional[List[float]]: List of closest distances between the bounding boxes in bounds1 and bounds2.
    """
    bounds1, bounds2 = row[0], row[1]
    if not bounds1 or not bounds2:
        return None
    
    closest_distances = []
    for bound1 in bounds1:
        if bound1 is None or not len(bound1):
            continue
        corners1 = np.array(bound1)
        min_distance = min(
            cdist(corners1, np.array(bound2)).min()
            for bound2 in bounds2 if bound2 is not None and len(bound2)
        )
        closest_distances.append(min_distance)
    
    return closest_distances

def shortest_dist(bounds1: pd.DataFrame, bounds2: pd.DataFrame) -> pd.Series:
    """
    Calculate the shortest distances between two sets of bounding boxes.

    Args:
    - bounds1 (pd.DataFrame): DataFrame containing the first set of bounding boxes.
    - bounds2 (pd.DataFrame): DataFrame containing the second set of bounding boxes.

    Returns:
    - pd.Series: Series containing the shortest distances between the bounding boxes.
    """
    merged = pd.merge(bounds1, bounds2, left_index=True, right_index=True)
    return merged.apply(closest_distances, axis=1)

def min_distance(row: pd.Series) -> float:
    """
    Calculate the minimum distance between vegetation and house points.

    Args:
    - row (pd.Series): Series containing vegetation and house points.

    Returns:
    - float: The minimum distance between the vegetation and house points.
    """
    veg_points = np.array(row['vegetation.'])
    house_points = np.array(row['house.'])
    distances = cdist(veg_points, house_points)
    return distances.min()

def dist(bounds1: pd.DataFrame, bounds2: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the minimum distances between two sets of bounding boxes.

    Args:
    - bounds1 (pd.DataFrame): DataFrame containing the first set of bounding boxes.
    - bounds2 (pd.DataFrame): DataFrame containing the second set of bounding boxes.

    Returns:
    - pd.DataFrame: DataFrame containing the minimum distances.
    """
    merged = bounds1.merge(bounds2, left_index=True, right_index=True).explode(bounds1.name).explode(bounds2.name)
    merged['min_distance'] = merged.apply(min_distance, axis=1)
    
    return merged

def dist_by_obj(dist: pd.DataFrame, label: str, fn=min) -> pd.DataFrame:
    """
    Group distances by object and apply a function to the minimum distances.

    Args:
    - dist (pd.DataFrame): DataFrame containing the distances.
    - label (str): Column name to group by.
    - fn (function): Function to apply to the distances.

    Returns:
    - pd.DataFrame: DataFrame containing the grouped distances.
    """
    dist[label] = dist[label].apply(tuple)
    return dist.groupby(['original_index', label])['min_distance'].agg(fn).reset_index().groupby('original_index')['min_distance'].agg(list)

def nearest_obj_exist(
    gdf: gpd.GeoDataFrame, objs: gpd.GeoDataFrame, meta: pd.DataFrame, 
    max_dist: float = 0.001, parcels: Optional[gpd.GeoDataFrame] = None, 
    visualize: bool = False
) -> pd.Series:
    """
    Check if nearest objects exist within a specified distance.

    Args:
    - gdf (gpd.GeoDataFrame): GeoDataFrame containing geometries.
    - objs (gpd.GeoDataFrame): GeoDataFrame containing objects.
    - meta (pd.DataFrame): DataFrame containing metadata.
    - max_dist (float): Maximum distance to consider for nearest objects.
    - parcels (Optional[gpd.GeoDataFrame]): GeoDataFrame containing parcel geometries.
    - visualize (bool): Whether to visualize the results.

    Returns:
    - pd.Series: Series indicating the existence of nearest objects.
    """
    obj_meta = pd.merge(objs, meta['meta_pt'][~meta['meta_pt'].index.duplicated(keep='first')], left_index=True, right_index=True)
    obj_meta = gpd.GeoDataFrame(obj_meta).set_geometry('meta_pt')
    joined = gpd.sjoin_nearest(gdf, obj_meta, how='left', max_distance=max_dist)
    nearest_exist = ~joined['index_right'].isna()
    
    if visualize: 
        fig, ax = plt.subplots(figsize=(8, 8))

        gdf.plot(ax=ax)
        obj_meta.plot(ax=ax)
        true_index = gdf[nearest_exist].index[0]
        true_point = gdf.loc[true_index, 'geometry']
        
        circle = Circle((true_point.x, true_point.y), 0.001, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        parcels.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1)
    return 

def geoloc_est(bounds: List[np.ndarray], img_loc: Point, heading: float, lat_scale: float = 111320) -> List[Point]:
    """
    Estimate geographic locations from bounding boxes.

    Args:
    - bounds (List[np.ndarray]): List of bounding boxes.
    - img_loc (Point): Image location.
    - heading (float): Heading angle.
    - lat_scale (float): Latitude scale.

    Returns:
    - List[Point]: List of estimated geographic locations.
    """
    geolocs = []

    lon, lat = img_loc.x, img_loc.y
    lon_scale = lat_scale * np.cos(np.radians(lat))
    
    for bd in bounds: 
        x, y, z = np.mean(bd, axis=0)
        heading_rad = math.radians(heading)
        
        delta_lat = z / lat_scale * np.cos(heading_rad) - x / lon_scale * np.sin(heading_rad)
        delta_lon = x / lon_scale * np.cos(heading_rad) + z / lat_scale * np.sin(heading_rad)
           
        geolocs.append(Point(lon + delta_lon, lat + delta_lat))
        
    return geolocs
    
def new_pos(point: Point, heading: float, distance: float) -> Point:
    """
    Calculate a new position from a point given a heading and distance.

    Args:
    - point (Point): Original point.
    - heading (float): Heading angle.
    - distance (float): Distance to move.

    Returns:
    - Point: New position.
    """
    heading_rad = math.radians(heading)
    delta_y = distance * math.sin(heading_rad)
    delta_x = distance * math.cos(heading_rad)
    return Point(point.x + delta_x, point.y + delta_y)

def geoloc_est_obj(bounds: Dict[str, List[np.ndarray]], headings: pd.Series, img_locs: pd.Series) -> Dict[str, gpd.GeoSeries]:
    """
    Estimate geographic locations for each object.

    Args:
    - bounds (Dict[str, List[np.ndarray]]): Dictionary of bounding boxes grouped by label.
    - headings (pd.Series): Series of headings.
    - img_locs (pd.Series): Series of image locations.

    Returns:
    - Dict[str, gpd.GeoSeries]: Dictionary of GeoSeries containing estimated geographic locations.
    """
    results = {}
    for label, bds in bounds.items():
        geolocs = []
        for i, bd in enumerate(bds):
            geolocs.extend(geoloc_est(bd, headings.iloc[i], img_locs.iloc[i]))
        results[label] = gpd.GeoSeries(geolocs)
    return results