import cv2
import numpy as np

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


def normalize_image(image: np.ndarray, min_value: float = 0, max_value: float = 1000) -> np.ndarray:
    delta = max_value - min_value

    img = image.astype(np.float32).clip(min_value, max_value)
    img = (((img - min_value) / delta) * 255).astype(np.uint8)
    return img
