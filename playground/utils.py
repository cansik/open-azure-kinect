from typing import Tuple

import cv2
import numpy as np

from playground.charuco_utils import MarkerDetectionResult

COLOR_SEQUENCE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (255, 255, 255),
]


def annotate_points(image: np.ndarray, points: np.ndarray, marker_type: int = cv2.MARKER_CROSS):
    h, w = image.shape[:2]

    factor = (h * w) / (640 * 576)

    size = max(1, round(10 * factor))
    thickness = min(4, max(1, round(2 * factor)))

    image_points = np.round(points).astype(np.int32)
    for i, point in enumerate(image_points):
        color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
        cv2.drawMarker(image, point, color, markerType=marker_type, markerSize=size, thickness=thickness)


def concat_images_horizontally(*images: np.ndarray, target_height: int = 480):
    resized_images = []

    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized_img)

    concatenated_image = np.hstack(resized_images)
    return concatenated_image


def concat_images_vertically(*images: np.ndarray, target_width: int = 1920):
    resized_images = []

    for img in images:
        aspect_ratio = img.shape[0] / img.shape[1]
        new_height = int(target_width * aspect_ratio)
        resized_img = cv2.resize(img, (target_width, new_height))
        resized_images.append(resized_img)

    concatenated_image = np.vstack(resized_images)
    return concatenated_image


def normalize_image(image: np.ndarray, min_value: float = 0, max_value: float = 2000) -> np.ndarray:
    delta = max_value - min_value

    img = image.astype(np.float32).clip(min_value, max_value)
    img = (((img - min_value) / delta) * 255).astype(np.uint8)
    return img


def compute_projection_error(
        source_markers: MarkerDetectionResult,
        destination_markers: MarkerDetectionResult,
        estimated_destination_points: np.ndarray
) -> Tuple[int, float]:
    """
    Given a set of source-detected markers and their IDs, the corresponding
    estimated points in the destination image, and the actual destination
    detections, compute the mean reprojection error.

    Args:
        source_markers: MarkerDetectionResult for the source image
        destination_markers: MarkerDetectionResult for the destination image
        estimated_destination_points: (N,2) ndarray of projected source points
            into the destination image; N == number of source_markers.ids

    Returns:
        count: number of markers used in the error computation
        mean_error: mean Euclidean pixel error over the matched markers
    """
    # Build dict of actual destination centers: id -> (x,y)
    dest_centers = {}
    if destination_markers.ids is not None:
        for corners, mid in zip(destination_markers.corners, destination_markers.ids.flatten()):
            pts = corners.reshape(-1, 2)  # flatten (4,1,2) -> (4,2)
            dest_centers[int(mid)] = pts.mean(axis=0)

    errors = []
    if source_markers.ids is not None:
        for idx, mid in enumerate(source_markers.ids.flatten()):
            if mid in dest_centers:
                est_pt = estimated_destination_points[idx]
                true_pt = dest_centers[int(mid)]
                errors.append(np.linalg.norm(est_pt - true_pt))

    count = len(errors)
    mean_error = float(np.mean(errors)) if count > 0 else float('nan')
    return count, mean_error
