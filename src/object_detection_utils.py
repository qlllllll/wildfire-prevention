import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import itertools
import os
import requests
import sys
from contextlib import contextmanager

import cv2
import torch
from torchvision.ops import box_convert
import torchvision.ops as ops

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class BoundingBox:
    """
    A class to represent a bounding box with its coordinates.

    Attributes:
    - xmin (int): The minimum x-coordinate of the bounding box.
    - ymin (int): The minimum y-coordinate of the bounding box.
    - xmax (int): The maximum x-coordinate of the bounding box.
    - ymax (int): The maximum y-coordinate of the bounding box.
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        """
        Get the bounding box coordinates in [xmin, ymin, xmax, ymax] format.

        Returns:
        - List[float]: The coordinates of the bounding box.
        """
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    """
    A class to represent the result of an object detection.

    Attributes:
    - score (float): The confidence score of the detection.
    - label (str): The label or class of the detected object.
    - box (BoundingBox): The bounding box of the detected object.
    - mask (Optional[np.ndarray]): The segmentation mask of the detected object, if available.
    """
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None  

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        """
        Create a DetectionResult instance from a dictionary.

        Args:
        - detection_dict (Dict): A dictionary containing detection result information.

        Returns:
        - DetectionResult: An instance of the DetectionResult class.
        """
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'][:-1],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Convert a binary mask to a polygon representation.

    Args:
    - mask (np.ndarray): A binary mask where the object is represented by non-zero values.

    Returns:
    - List[List[int]]: A list of [x, y] coordinates representing the vertices of the polygon.
    """
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
    - polygon (List[Tuple[int, int]]): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (Tuple[int, int]): Shape of the image (height, width) for the mask.

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

def load_images(folder_path: str = './gsv_images', name: str = 'gsv') -> pd.Series:
    """
    Load images from a specified folder into a pandas Series.

    Args:
    - folder_path (str): The path to the folder containing the images. Default is './gsv_images'.
    - name (str): The name of the pandas Series. Default is 'gsv'.

    Returns:
    - pd.Series: A pandas Series containing the loaded images.
    """
    
    def open_img(file_path: str) -> Image.Image:
        with Image.open(file_path) as img:
            return img.copy()

    image_data = [open_img(os.path.join(folder_path, file_name)) for file_name in sorted(os.listdir(folder_path)) if file_name.endswith(('.png', '.jpg', '.jpeg'))]
    return pd.Series(image_data, name=name)

def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]: 
    """
    Extract bounding box coordinates from a list of detection results.

    Args:
    - results (List[DetectionResult]): A list of detection results.

    Returns:
    - List[List[List[float]]]: A nested list containing bounding box coordinates in [xmin, ymin, xmax, ymax] format.
    """
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    """
    Refine segmentation masks, optionally applying polygon refinement.

    Args:
    - masks (torch.BoolTensor): A tensor containing the masks.
    - polygon_refinement (bool): Whether to apply polygon refinement to the masks. Default is False.

    Returns:
    - List[np.ndarray]: A list of refined masks.
    """
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
    detector_id: Optional[str] = None,
    iou_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.

    Args:
    - image (Image.Image): Input image.
    - labels (List[str]): List of labels to detect.
    - threshold (float): Detection threshold.
    - detector_id (Optional[str]): Detector model ID.
    - iou_threshold (float): Intersection-over-Union threshold for non-max suppression.

    Returns:
    - List[Dict[str, Any]]: Detection results.
    """
    if isinstance(image, str):
        image = load_image(image)
        
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=DEVICE)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    detections = object_detector(image, candidate_labels=labels, threshold=threshold)
    
    boxes, scores = [], []

    for detection in detections:
        box = detection['box']
        boxes.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
        scores.append(detection['score'])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

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
) -> List[DetectionResult]:  
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
    - List[DetectionResult]: Segmentation results.
    """
    if isinstance(image, str):
        image = load_image(image)

    with suppress_output():
        detections = detect(image, labels, threshold, detector_id)
        detections = segment(image, detections, polygon_refinement, segmenter_id)

    return detections

def object_grounded_segmentation(image_series: pd.Series, text_prompt: List[str]) -> pd.Series:
    """
    Perform object detection and segmentation on a series of images using a text prompt.

    Args:
    - image_series (pd.Series): Series of images.
    - text_prompt (List[str]): List of text prompts for detection.

    Returns:
    - pd.Series: Series of detection and segmentation results.
    """
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    
    detections = image_series.apply(lambda img: grounded_segmentation(image=img.image, labels=text_prompt, threshold=0.3, polygon_refinement=True, detector_id=detector_id, segmenter_id=segmenter_id))

    return detections 

def reformat_detections(detections: pd.Series) -> pd.DataFrame:
    """
    Convert a series of detection results into a structured DataFrame.

    Args:
    - detections (pd.Series): Series where each element is a list of DetectionResult objects.

    Returns:
    - pd.DataFrame: DataFrame with columns for image index, label, mask, bounding box coordinates (box), score, and a unique object index for each detection grouped by image index and label.
    """
    df = detections.to_frame('detections')

    df = df.explode('detections')

    df['image_index'] = df.index
    df['label'] = df['detections'].apply(lambda x: x.label)
    df['mask'] = df['detections'].apply(lambda x: x.mask)
    df['box'] = df['detections'].apply(lambda x: x.box.xyxy)
    df['score'] = df['detections'].apply(lambda x: x.score)
    
    df = df.sort_values(by=['image_index', 'label'])
    df.reset_index(inplace=True, drop=True)
    df.drop(columns=['detections'], inplace=True)
    df['object_index'] = df.groupby(['image_index', 'label']).cumcount()
    
    return df

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
    """
    Convert a depth map to 3D points.

    Args:
    - depth (np.ndarray): The depth map.
    - R (np.ndarray, optional): Rotation matrix. Defaults to identity matrix.
    - t (np.ndarray, optional): Translation vector. Defaults to zero vector.

    Returns:
    - np.ndarray: The 3D points.
    """
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

@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def depth_estimate(img_series: pd.Series) -> pd.Series:
    """
    Estimate depth from a series of images.

    Args:
    - img_series (pd.Series): Series of images.

    Returns:
    - pd.Series: Series of depth maps.
    """
    repo = "isl-org/ZoeDepth"
    
    with suppress_output():
        model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    
    zoe = model_zoe_k.to(DEVICE)
    return img_series.apply(lambda img: zoe.infer_pil(img.image))

def convert_depth_to_coords(depth_series: pd.Series, est_func=depth_to_points) -> pd.Series:
    """
    Convert depth maps to 3D coordinates.

    Args:
    - depth_series (pd.Series): Series of depth maps.
    - est_func (callable): Function to convert depth maps to 3D points.

    Returns:
    - pd.Series: Series of 3D coordinates for each depth map.
    """
    return depth_series.apply(est_func)

def calculate_3d_bounding_boxes(row: pd.Series) -> Optional[List[List[Tuple[float, float, float]]]]:
    """
    Calculate 3D bounding boxes from masks and coordinates.

    Args:
    - row (pd.Series): A series containing masks and coordinates.

    Returns:
    - Optional[List[List[Tuple[float, float, float]]]]: List of 3D bounding boxes.
    """
    mask, coords = row['mask'], row['gsv']
    if mask is None:
        return None 
    
    corners = tuple(generate_box_corners(
                np.concatenate((
                np.min(image_3d_coords := coords[mask.flatten() == 255], axis=0),
                np.max(image_3d_coords, axis=0)
            ))
        ))

    return corners

def generate_3d_bounding_boxes(mask_series: pd.DataFrame, coords_series: pd.Series) -> pd.Series:
    """
    Generate 3D bounding boxes for each object.

    Args:
    - mask_series (pd.DataFrame): DataFrame containing the masks.
    - coords_series (pd.Series): Series containing the coordinates.

    Returns:
    - pd.Series: Series containing the 3D bounding boxes, with the name set to the label.
    """
    merged = pd.merge(mask_series, coords_series, left_on='image_index', right_index=True)
    bounds = merged.apply(calculate_3d_bounding_boxes, axis=1)

    return bounds

def generate_box_corners(bounds: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Generate corners of a bounding box.

    Args:
    - bounds (np.ndarray): Bounding box coordinates.

    Returns:
    - List[Tuple[float, float, float]]: List of corners.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return list(itertools.product([min_x, max_x], [min_y, max_y], [min_z, max_z]))

def stat_distance(row: pd.Series, label1: str, label2: str, fn: callable = np.min) -> float:
    """
    Calculate a specified statistical measure (e.g., minimum, maximum) of distances between two sets of points.

    Args:
    - row (pd.Series): Series containing two sets of points.
    - label1 (str): Label for the first set of points.
    - label2 (str): Label for the second set of points.
    - fn (Callable): Function to apply to the calculated distances (e.g., min, max). Default is min.

    Returns:
    - float: The result of applying the specified function to the distances between the two sets of points.
    """
    pts1 = np.array(row[f"coords_{label1}"])
    pts2 = np.array(row[f"coords_{label2}"])
    distances = cdist(pts1, pts2)
    return fn(distances)

def dist(bounds: pd.DataFrame, label1: str, label2: str, fn: callable = np.min) -> pd.DataFrame:
    """
    Calculate the distances between two sets of bounding boxes.

    Args:
    - bounds1 (pd.Series): Series containing the first set of bounding boxes.
    - bounds2 (pd.Series): Series containing the second set of bounding boxes.
    - label1 (str): Label for the first set of bounding boxes.
    - label2 (str): Label for the second set of bounding boxes.
    - fn (Callable): Function to apply to the calculated distances (e.g., min, max). Default is min.

    Returns:
    - pd.DataFrame: DataFrame containing the distances.
    """
    bounds1 = bounds[bounds['label']==label1].drop(columns=['label'])
    bounds2 = bounds[bounds['label']==label2].drop(columns=['label'])
    merged = pd.merge(bounds1, bounds2, left_on='image_index', right_on='image_index', suffixes=(f'_{label1}', f'_{label2}'))
    merged['distance'] = merged.apply(lambda row: stat_distance(row, label1, label2, fn), axis=1)
    
    return merged

def group_distances(distances: pd.DataFrame, detects: pd.DataFrame, label1: str, label2: str, fn: callable = min) -> pd.DataFrame:
    """
    Group distances by object and apply a function to the distances.

    Args:
    - distances (pd.DataFrame): DataFrame containing the distances.
    - detects (pd.DataFrame): DataFrame containing the detections.
    - label1 (str): Label for the first set of detections.
    - label2 (str): Label for the second set of detections.
    - fn (Callable): Function to apply to the distances. Default is min.
    - thres (float): Threshold for distance grouping.

    Returns:
    - pd.DataFrame: DataFrame containing the grouped distances.
    """
    grouped_distances = distances.groupby(['image_index', f"coords_{label1}"]).agg(fn).reset_index()
    merged_df = grouped_distances.merge(detects[['mask', 'box', 'score', 'object_index', 'coords']], left_on=f"coords_{label1}", right_on='coords').drop(columns=['coords'])
    merged_df = merged_df.merge(detects[['mask', 'box', 'score', 'object_index', 'coords']], left_on=f"coords_{label2}", right_on='coords', suffixes=(f'_{label1}', f'_{label2}')).drop(columns=['coords'])
    return merged_df   

def nearest_object_existence(
    gdf: gpd.GeoDataFrame, objs: gpd.GeoDataFrame, meta: pd.DataFrame, 
    max_dist: float = 0.001, visualize: bool = False
) -> pd.Series:
    """
    Check if nearest objects exist within a specified distance.

    Args:
    - gdf (gpd.GeoDataFrame): GeoDataFrame containing geometries.
    - objs (gpd.GeoDataFrame): GeoDataFrame containing objects.
    - meta (pd.DataFrame): DataFrame containing metadata.
    - max_dist (float): Maximum distance to consider for nearest objects.
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
        gpd.GeoDataFrame(meta).plot(ax=ax, linewidth=1)
    return nearest_exist

def estimate_geolocations(bounds: List[Tuple[float, float, float]], img_loc: Point, heading: float, lat_scale: float = 111320) -> List[Point]:
    """
    Estimate geographic locations from bounding boxes.

    Args:
    - bounds (List[Tuple[float, float, float]]): List of tuples, where each tuple represents a point in the bounding box as (x, y, z).
    - img_loc (Point): Geographic location of the image (longitude, latitude).
    - heading (float): Heading angle in degrees.
    - lat_scale (float): Scaling factor for latitude to meters. Default is 111320.

    Returns:
    - List[Point]: List of estimated geographic locations as shapely Point objects.
    """
    lon, lat = img_loc.x, img_loc.y
    lon_scale = lat_scale * np.cos(np.radians(lat))

    x, y, z = np.mean(bounds, axis=0)
    heading_rad = math.radians(heading)
    
    delta_lat = z / lat_scale * np.cos(heading_rad) - x / lon_scale * np.sin(heading_rad)
    delta_lon = x / lon_scale * np.cos(heading_rad) + z / lat_scale * np.sin(heading_rad)
    
    return Point(lon + delta_lon, lat + delta_lat)

def estimate_object_locations(bounds: pd.DataFrame, meta: pd.DataFrame, visualize: bool = False) -> gpd.GeoSeries:
    """
    Estimate geographic locations for each object.

    Args:
    - bounds (pd.DataFrame): DataFrame of bounding boxes grouped by label.
    - meta (pd.DataFrame): DataFrame containing metadata.
    - visualize (bool): Whether to visualize the results.

    Returns:
    - gpd.GeoSeries: GeoSeries containing estimated geographic locations.
    """
    bounds = bounds.merge(meta[['meta_pt', 'heading']], left_on='image_index', right_index=True)
    geolocs = bounds.apply(lambda row: estimate_geolocations(row['coords'], row['meta_pt'], row['heading']), axis=1)

    if visualize:
        fig, ax = plt.subplots(figsize=(8, 8))
        unique_labels = bounds['label'].unique()
        cmap = get_cmap('tab10') 

        for idx, label in enumerate(unique_labels):
            label_geolocs = gpd.GeoSeries([geoloc for geoloc, lbl in zip(geolocs, bounds['label']) if lbl == label])
            label_geolocs.plot(ax=ax, markersize=5, color=cmap(idx), label=label)

        gpd.GeoSeries(meta['geometry']).plot(ax=ax, markersize=5)
        plt.legend()
        plt.show()

    return gpd.GeoSeries(geolocs)
    
def new_position(point: Point, heading: float, distance: float) -> Point:
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

