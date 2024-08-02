# Object Detection on Google Street View Data

This Python module performs object detection on Google Street View photos fetched from specified geolocation boundaries. It allows downloading Google Street View images at evenly spaced points along the road network within the boundary. The module uses Grounding-SAM for detecting and segmenting objects based on text prompts from the images. Additionally, it estimates distances from image metadata geolocation using Zoe-Depth and includes custom functions to analyze the geospatial relationships between objects.

### Install

Install from git with:
```bash
git clone https://github.com/qlllllll/fire-risk-object-detection.git
pip install -r requirements.txt
```

### Quick Start

#### Downloading Google Street View Images of an Area

To download Google Street View images for a specific area, follow these steps:

1. Use `get_area_bbox` to access the Google Maps API and retrieve the bounding box for the specified area.
2. Use `generate_network_pts` to access the road network from OpenStreetMap using the geo boundary, and generate sample points along the network.
3. Use `load_gsv_img_from_coords` to load Google Street View images using the generated coordinates and save them in the specified directory.

```python
from geo_utils import get_area_bbox, generate_network_pts, load_gsv_img_from_coords

# Define the area of interest
area = 'Northside, Berkeley, CA'

# Retrieve the bounding box for the specified area
bbox = get_area_bbox(area)

# Generate sample points along the road network within the bounding box
sample_points = generate_network_pts(bbox, samp_dist=0.00015)

# Download Google Street View images using the generated sample points
load_gsv_images(sample_points, save_dir='gsv_images')
```

#### Object Detection on Series of Images

For object detection on a series of images loaded from a folder, the following functions operate on `pd.Series` for better readability and flow:

- `depth_estimate`: Returns a series of depth maps approximated with Zoe-Depth.
- `convert_depth_to_coords`: Projects pixels onto a 3D plane using a sample camera intrinsic matrix.
- `object_grounded_segmentation`: Returns a series of `DetectionResult` objects, which include [score, label, box, mask] from Grounding-SAM.
- `reformat_detections`: Returns a dictionary of `pd.Series` grouped by label.
- `generate_3d_bounding_boxes`: Estimates the corner coordinates of the 3D bounding box for the object series.

```python
from object_detection_utils import load_images, depth_estimate, convert_depth_to_coords, object_grounded_segmentation, generate_3d_bounding_boxes, reformat_detections

# Load your images into a pd.Series
images = load_images(folder_path='./gsv_images')[:10]

# Estimate depth maps for the images
depth_maps = depth_estimate(images)

# Project pixels onto a 3D plane
coords = convert_depth_to_coords(depth_maps)

# Perform object detection
detections = object_grounded_segmentation(images, ['vegetation', 'house', 'fire hydrant'])

# Reformat detections into a dictionary grouped by label
detection_dict = reformat_detections(detections)

# Estimate 3D bounding boxes for specific objects
bounds_vegetation = generate_3d_bounding_boxes(detection_dict['vegetation'], coords, 'vegetation')
bounds_house = generate_3d_bounding_boxes(detection_dict['house'], coords, 'house')
```

#### Geospatial Relationships between Objects
To analyze the geospatial relationships between detected objects, use the following functions:

- `dist`: Calculates the minimum distances between two sets of bounding boxes.
- `group_distances`: Groups distances by object and applies a specified function to the minimum distances.

```python
from object_detection_utils import dist, group_distances

# Calculate the minimum distances between vegetation and house bounding boxes
distances = dist(bounds_vegetation, bounds_house, np.min)

# Group distances by object (house) and find the minimum distance to vegetation
dist_house_to_veg_min = group_distances(distances, 'house', np.min)
```

To check if the nearest objects exist within a specified distance, use the `nearest_object_existence` function. This function can also visualize the results if needed.

```python
from object_detection_utils import nearest_object_existence

# Check if fire hydrants exist within 0.001 distance from sample hydrants map
sample_hydrants = ...
nearest_exist = nearest_object_existence(sample_hydrants, detection_dict['fire hydrant'], meta=sample_points, max_dist=0.001, visualize=True)
```

To estimate the geographic locations of objects from bounding boxes, use the `estimate_object_locations` function. These functions can also support visualizing the estimated locations.

```python
from object_detection_utils import estimate_object_locations

# Estimate geographic locations for the detected objects
geoloc_results = estimate_object_locations(bounds_vegetation, meta=sample_points, visualize=True)
```
