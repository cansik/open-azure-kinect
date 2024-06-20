import argparse
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from openk4a.calibration import CameraCalibration
from openk4a.playback import OpenK4APlayback
from playground.utils import annotate_points


@dataclass
class DistortionMapping:
    x_mapping: np.ndarray
    y_mapping: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        points_int32 = points.astype(np.int32)
        x_values = np.array([self.x_mapping[y, x] for x, y in points_int32])
        y_values = np.array([self.y_mapping[y, x] for x, y in points_int32])

        return np.hstack((x_values.reshape(-1, 1), y_values.reshape(-1, 1)))

    def remap(self, image: np.ndarray, interpolation_method: int = cv2.INTER_NEAREST):
        return cv2.remap(image, self.x_mapping, self.y_mapping, interpolation_method)


def compute_distortion_mapping(calibration: CameraCalibration) -> DistortionMapping:
    x_map, y_map = cv2.initUndistortRectifyMap(
        calibration.intrinsics.camera_matrix,
        calibration.intrinsics.distortion_coefficients,
        np.eye(3),
        calibration.intrinsics.camera_matrix,
        (calibration.width, calibration.height),
        5,  # CV_32FC1
    )

    return DistortionMapping(x_map, y_map)


def compute_inverse_distortion_mapping(calibration: CameraCalibration) -> DistortionMapping:
    x_map, y_map = cv2.initInverseRectificationMap(
        calibration.intrinsics.camera_matrix,
        calibration.intrinsics.distortion_coefficients,
        np.eye(3),
        calibration.intrinsics.camera_matrix,
        (calibration.width, calibration.height),
        5,  # CV_32FC1
    )

    return DistortionMapping(x_map, y_map)


def concat_images_horizontally(*images: np.ndarray, target_height: int = 480):
    resized_images = []

    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized_img)

    concatenated_image = np.hstack(resized_images)
    return concatenated_image


def predict_face_bounding_box(frame: np.ndarray) -> Optional[np.ndarray]:
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    if len(face) == 0:
        return None

    x, y, w, h = face[0]
    return np.array([x, y, x + w, y + h], dtype=np.float32).reshape(2, 2)


def normalize_image(image: np.ndarray, min_value: float = 0, max_value: float = 1000) -> np.ndarray:
    delta = max_value - min_value

    img = image.astype(np.float32).clip(min_value, max_value)
    img = (((img - min_value) / delta) * 255).astype(np.uint8)
    return img


def calculate_homography(K1: np.ndarray, K2: np.ndarray, R: np.ndarray, t: np.ndarray, Z0: float) -> np.ndarray:
    """
    Calculate the homography matrix between two cameras.

    Args:
        K1 (np.ndarray): Intrinsic matrix of the first camera.
        K2 (np.ndarray): Intrinsic matrix of the second camera.
        R (np.ndarray): Rotation matrix from the first camera to the second camera.
        t (np.ndarray): Translation vector from the first camera to the second camera.
        Z0 (float): Depth of the plane from the first camera.

    Returns:
        np.ndarray: Homography matrix.
    """
    # Normal to the plane
    n = np.array([0, 0, 1])

    # Homography calculation
    H = K2 @ (R - (t[:, None] @ n[None, :]) / Z0) @ np.linalg.inv(K1)

    return H


def transform_points_with_cv(points: np.ndarray,
                             calib_a: CameraCalibration,
                             calib_b: CameraCalibration,
                             Z0: float) -> np.ndarray:
    # Extract intrinsics
    K_a = calib_a.intrinsics.camera_matrix
    K_b = calib_b.intrinsics.camera_matrix

    # Extract extrinsics
    R_a_to_b = calib_b.extrinsics.rotation @ calib_a.extrinsics.rotation.T
    T_a_to_b = (calib_b.extrinsics.translation[0] * 1000) - R_a_to_b @ (calib_a.extrinsics.translation[0] * 1000)

    # Invert intrinsic matrix of camera A
    K_a_inv = np.linalg.inv(K_a)

    # Convert points to homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Normalize points (from image coordinates to camera coordinates)
    points_cam_a = K_a_inv @ points_h.T

    # Assume Z0 depth and convert normalized points to 3D points
    points_3d_a = points_cam_a * Z0

    # Transform points from camera A coordinate system to camera B coordinate system
    points_3d_b = (R_a_to_b @ points_3d_a) + T_a_to_b[:, np.newaxis]

    # Project 3D points to image plane of camera B
    points_h_b = K_b @ points_3d_b

    # Normalize the points
    points_b = points_h_b[:2, :] / points_h_b[2, :]

    return points_b.T


def transform_points_between_cameras_using_homography(points: np.ndarray,
                                                      calib_a: CameraCalibration,
                                                      calib_b: CameraCalibration,
                                                      Z0: float) -> np.ndarray:
    """
    Transform points from one camera to another using the homography matrix, taking distortion into account.

    Args:
        points (np.ndarray): Array of 2D points (Nx2) in the first camera's image plane.
        calib_a (CameraCalibration): Calibration information for the first camera.
        calib_b (CameraCalibration): Calibration information for the second camera.
        Z0 (float): Depth of the plane from the first camera.

    Returns:
        np.ndarray: Transformed 2D points (Nx2) in the second camera's image plane.
    """
    K1 = calib_a.intrinsics.camera_matrix
    D1 = calib_a.intrinsics.distortion_coefficients
    K2 = calib_b.intrinsics.camera_matrix
    D2 = calib_b.intrinsics.distortion_coefficients

    # Compute relative rotation and translation
    R_A = calib_a.extrinsics.rotation
    t_A = calib_a.extrinsics.translation
    R_B = calib_b.extrinsics.rotation
    t_B = calib_b.extrinsics.translation

    R_BA = R_B @ np.linalg.inv(R_A)
    t_BA = t_B - R_BA @ t_A

    # Calculate the homography matrix
    H = calculate_homography(K1, K2, R_BA, t_BA, Z0)

    # Convert undistorted points to homogeneous coordinates
    undistorted_points_h = np.hstack([points, np.ones((points.shape[0], 1))])

    # Transform points using the homography matrix
    transformed_points_h = (H @ undistorted_points_h.T).T

    # Convert back to non-homogeneous coordinates
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2][:, np.newaxis]

    return transformed_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = True
    azure.open()

    points_first_frame = np.array([
        [960, 540],
        [694, 444],
        [746, 553],
        [1096, 514],
        [968, 305],

        [975, 529],
        [976, 482],
        [975, 437],
        [974, 394],
        [974, 394],
        [974, 350],
    ], dtype=np.float32)

    while capture := azure.read():
        preview_color = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
        preview_infrared = cv2.cvtColor(normalize_image(capture.ir), cv2.COLOR_GRAY2BGR)

        color_distortion_mapping = compute_distortion_mapping(azure.color_calibration)
        inv_color_distortion_mapping = compute_inverse_distortion_mapping(azure.color_calibration)
        infrared_distortion_mapping = compute_distortion_mapping(azure.depth_calibration)

        preview_color = color_distortion_mapping.remap(preview_color)
        preview_infrared = infrared_distortion_mapping.remap(preview_infrared)

        # face_bbox = predict_face_bounding_box(preview_color)
        face_bbox = inv_color_distortion_mapping.transform(points_first_frame)

        distance = capture.depth[576 // 2, 640 // 2]

        if face_bbox is not None:
            infrared_face_bbox = transform_points_between_cameras_using_homography(face_bbox,
                                                                                   azure.color_calibration,
                                                                                   azure.depth_calibration, 1)
            infrared_face_bbox = transform_points_with_cv(face_bbox, azure.color_calibration, azure.depth_calibration,
                                                          distance / 1000)

            # annotate infrared image
            annotate_points(preview_color, face_bbox)
            annotate_points(preview_infrared, infrared_face_bbox)
            # cv2.rectangle(preview_infrared, infrared_face_bbox[0], infrared_face_bbox[1], (0, 255, 0), 4)

        cv2.imshow("Result", concat_images_horizontally(preview_color, preview_infrared))
        cv2.waitKey(0)

    azure.close()


if __name__ == "__main__":
    main()
