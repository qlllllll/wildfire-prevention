import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Union, List, Optional, Dict
from PIL import Image
import webcolors

from object_detection_utils import *

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    """
    Annotate image with bounding boxes and masks for detection results.

    Args:
    - image (Union[Image.Image, np.ndarray]): The image to be annotated.
    - detection_results (List[DetectionResult]): List of detection results.

    Returns:
    - np.ndarray: Annotated image in numpy array format.
    """
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def annotate_mask_dist(image, row, label1, label2, save_name=None):
    """
    Annotate image with masks and distance information for specified labels.

    Args:
    - image (Union[Image.Image, np.ndarray]): The image to be annotated.
    - row (pd.Series): The row containing the data for annotation.
    - label1 (str): First label for annotation.
    - label2 (str): Second label for annotation.
    - save_name (Optional[str]): Path to save the annotated image. Defaults to None.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    padding = 15
    image_cv2 = cv2.copyMakeBorder(image_cv2, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    colors = random_named_css_colors(2)

    for index, lbl in enumerate([label1, label2]): 
        xmin, ymin, _, _ = row[f'box_{lbl}']
        xmin += padding
        ymin += padding
    
        mask = np.array(row[f'mask_{lbl}'])
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_name = colors[index]
        color = webcolors.name_to_rgb(color_name)
        color = (color.red, color.green, color.blue)
        cv2.drawContours(image_cv2, [c + [padding, padding] for c in contours], -1, color, 2)

        obj_idx = row[f"object_index_{lbl}"]
        text = f'{lbl} [{obj_idx}]: {row["distance"]:.5f}'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_cv2, (xmin + 5, ymin - text_height - baseline), 
                      (xmin + 5 + text_width, ymin + baseline), color, thickness=cv2.FILLED)
        cv2.putText(image_cv2, text, (xmin + 5, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    plt.imshow(image_cv2)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def estimate_locations(dists: pd.DataFrame, index: int, label1: str, label2: str) -> None:
    """
    Visualize the closest points between two sets of coordinates on the X-Z plane, linking them with lines
    and annotating their distances for a specific image index.

    Args:
    - dists (pd.DataFrame): DataFrame containing distances and coordinate information.
    - index (int): The specific image index to filter the DataFrame and visualize the results.
    - label1 (str): The label for the first set of coordinates.
    - label2 (str): The label for the second set of coordinates.

    Returns:
    - None
    """
    df = dists[dists['image_index'] == index]
    
    fig, ax = plt.subplots()
    all_coords = []
    annotations = []

    for _, row in df.iterrows():
        coords1 = np.array([[coord[0], coord[2]] for coord in row[f"coords_{label1}"]])
        coords2 = np.array([[coord[0], coord[2]] for coord in row[f"coords_{label2}"]])
        distance, obj_index1, obj_index2 = row['distance'], row[f"object_index_{label1}"], row[f"object_index_{label2}"]
        
        all_coords.extend(coords1)
        all_coords.extend(coords2)
        
        dist_matrix = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)
        min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        point1, point2 = coords1[min_dist_idx[0]], coords2[min_dist_idx[1]]

        ax.plot(*point1, 'go', label=label1 if _ == 0 else "")
        ax.plot(*point2, 'ro', label=label2 if _ == 0 else "")

        ax.plot(*zip(point1, point2), 'b--')

        ax.text(*point1, f'{obj_index1}', fontsize=12, color='green')
        ax.text(*point2, f'{obj_index2}', fontsize=12, color='red')

        mid_point = np.mean([point1, point2], axis=0)

        offset_y = 10
        for ann in annotations:
            if np.abs(ann[0] - mid_point[0]) < 0.1:  
                offset_y += 15  

        annotation = ax.annotate(f'{distance:.5f}', xy=mid_point, textcoords="offset points",
                                 xytext=(0, offset_y), ha='center', fontsize=10, color='blue',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=0, alpha=0))
        annotations.append((mid_point[0], mid_point[1] + offset_y))

    all_coords = np.array(all_coords)
    padding = 1
    ax.set_xlim(all_coords[:, 0].min() - padding, all_coords[:, 0].max() + padding)
    ax.set_ylim(all_coords[:, 1].min() - padding, all_coords[:, 1].max() + padding)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Closest Points on X-Z Plane')
    plt.axis('equal')
    plt.show()

def plot_results(pil_img, scores: List[float], labels: List[str], boxes: List[List[int]]) -> None:
    """
    Plots detection results on an image.

    Args:
    - pil_img (Image.Image): The image to be plotted.
    - scores (List[float]): List of scores for each detection.
    - labels (List[str]): List of labels for each detection.
    - boxes (List[List[int]]): List of bounding boxes for each detection.

    Returns:
    - None
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label_text = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label_text, fontsize=15,
                bbox=dict(facecolor='none', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    """
    Plots detections on an image.

    Args:
    - image (Union[Image.Image, np.ndarray]): The image to be plotted.
    - detections (List[DetectionResult]): List of detection results.
    - save_name (Optional[str]): Path to save the plotted image. Defaults to None.

    Returns:
    - None
    """
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
     
def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    colors = ['aqua', 'black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 
             'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen',
             'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
             'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 
             'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'goldenrod', 'gray', 
             'green', 'grey', 'hotpink', 'indianred', 'indigo', 'lawngreen', 'lightcoral', 'lightsalmon', 
             'lightseagreen',  'lightslategray', 'lightslategrey', 'lime', 'limegreen', 
             'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 
             'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navy', 'olive', 
             'olivedrab', 'orange', 'orangered', 'orchid', 'palevioletred', 'peru', 'plum', 'purple', 'red', 'rosybrown', 
             'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver', 'skyblue', 'slateblue', 
             'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'tomato', 'turquoise', 'violet', 'yellowgreen']

    return random.sample(colors, min(num_colors, len(colors)))
