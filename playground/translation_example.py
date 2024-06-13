import argparse
from typing import Optional

import cv2
import numpy as np

from openk4a.calibration import CameraCalibration
from openk4a.playback import OpenK4APlayback


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


def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray:
    """
    Undistort points using the camera matrix and distortion coefficients.

    Args:
        points (np.ndarray): Array of 2D points (Nx2) to be undistorted.
        camera_matrix (np.ndarray): Camera matrix (3x3).
        distortion_coefficients (np.ndarray): Distortion coefficients (1xN).

    Returns:
        np.ndarray: Array of undistorted 2D points (Nx2).
    """
    points = points.reshape(-1, 1, 2)  # Reshape for OpenCV function
    undistorted_points = cv2.undistortPoints(points, camera_matrix, distortion_coefficients, P=camera_matrix)
    undistorted_points = undistorted_points.reshape(-1, 2)  # Reshape back to Nx2
    return undistorted_points


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

    # Undistort the points in the first camera's image plane
    undistorted_points_a = undistort_points(points, K1, D1)

    # Calculate the homography matrix
    H = calculate_homography(K1, K2, R_BA, t_BA, Z0)

    # Convert undistorted points to homogeneous coordinates
    undistorted_points_h = np.hstack([undistorted_points_a, np.ones((undistorted_points_a.shape[0], 1))])

    # Transform points using the homography matrix
    transformed_points_h = (H @ undistorted_points_h.T).T

    # Convert back to non-homogeneous coordinates
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2][:, np.newaxis]

    # Distort the transformed points using the second camera's distortion coefficients
    transformed_points_distorted = cv2.undistortPoints(transformed_points, K2, D2, P=K2)
    transformed_points_distorted = transformed_points_distorted.reshape(-1, 2)

    return transformed_points_distorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = True
    azure.open()

    while capture := azure.read():
        face_bbox = predict_face_bounding_box(capture.color)

        preview_color = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
        preview_infrared = cv2.cvtColor(normalize_image(capture.ir), cv2.COLOR_GRAY2BGR)

        if face_bbox is not None:
            infrared_face_bbox = transform_points_between_cameras_using_homography(face_bbox,
                                                                                   azure.color_calibration,
                                                                                   azure.depth_calibration, 1)

            # annotate infrared image
            infrared_face_bbox = infrared_face_bbox.astype(np.uint32)
            cv2.rectangle(preview_infrared, infrared_face_bbox[0], infrared_face_bbox[1], (0, 255, 0), 4)

        cv2.imshow("Result", concat_images_horizontally(preview_color, preview_infrared))
        cv2.waitKey(1)

    azure.close()


if __name__ == "__main__":
    main()
