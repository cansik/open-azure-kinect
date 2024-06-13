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
    Transform points from one camera to another using the homography matrix.

    Args:
        points (np.ndarray): Array of 2D points (Nx2) in the first camera's image plane.
        calib_a (CameraCalibration): Calibration information for the first camera.
        calib_b (CameraCalibration): Calibration information for the second camera.
        Z0 (float): Depth of the plane from the first camera.

    Returns:
        np.ndarray: Transformed 2D points (Nx2) in the second camera's image plane.
    """
    K1 = calib_a.intrinsics.camera_matrix
    K2 = calib_b.intrinsics.camera_matrix
    R = calib_b.extrinsics.rotation
    t = calib_b.extrinsics.translation

    # Calculate the homography matrix
    H = calculate_homography(K1, K2, R, t, Z0)

    # Convert points to homogeneous coordinates
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])

    # Transform points using the homography matrix
    transformed_points_h = (H @ points_h.T).T

    # Convert back to non-homogeneous coordinates
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2][:, np.newaxis]

    return transformed_points


def transform_points_between_cameras(points: np.ndarray,
                                     calib_a: CameraCalibration,
                                     calib_b: CameraCalibration) -> np.ndarray:
    # Step 1: Undistort points from Camera A
    undistorted_points_a = cv2.undistortPoints(
        points,
        calib_a.intrinsics.camera_matrix,
        calib_a.intrinsics.distortion_coefficients,
        None,
        calib_a.intrinsics.camera_matrix
    )

    # Convert to homogeneous coordinates
    undistorted_points_a_hom = cv2.convertPointsToHomogeneous(undistorted_points_a)[:, 0, :]

    # Step 2: Transform points to Camera B's coordinate system
    # Compute the relative rotation and translation between the cameras
    R_a_to_b = np.dot(calib_b.extrinsics.rotation, calib_a.extrinsics.rotation.T)
    t_a_to_b = calib_b.extrinsics.translation - np.dot(R_a_to_b, calib_a.extrinsics.translation)

    # Apply the transformation
    points_in_cam_b = np.dot(R_a_to_b, undistorted_points_a_hom.T).T + t_a_to_b

    # Step 3: Project points into Camera B's image
    points_2d_b = cv2.projectPoints(
        points_in_cam_b,
        np.zeros((3, 1)),  # No rotation needed as points are already in Camera B's frame
        np.zeros((3, 1)),  # No translation needed
        calib_b.intrinsics.camera_matrix,
        calib_b.intrinsics.distortion_coefficients
    )[0]

    return points_2d_b.reshape(-1, 2)


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
