from dataclasses import dataclass

import cv2
import numpy as np

from openk4a.calibration import CameraCalibration


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


def compute_homography(K1: np.ndarray, K2: np.ndarray, R_rel: np.ndarray, t_rel: np.ndarray, n: np.ndarray,
                       d: float) -> np.ndarray:
    """
    Calculate the homography matrix H given the intrinsic and extrinsic parameters of two cameras,
    accounting for the tilt between the cameras.

    Parameters:
    - K1: Intrinsic matrix of the first camera (3x3 numpy array)
    - K2: Intrinsic matrix of the second camera (3x3 numpy array)
    - R_rel: Relative rotation matrix (3x3 numpy array)
    - t_rel: Relative translation vector (3x1 numpy array)
    - n: Normal vector to the plane (3x1 numpy array)
    - d: Distance from the origin to the plane along the normal (scalar)

    Returns:
    - H: Homography matrix (3x3 numpy array)
    """
    # Check if the input parameters are valid
    if not (K1.shape == (3, 3) and K2.shape == (3, 3)):
        raise ValueError("Intrinsic matrices K1 and K2 must be 3x3.")
    if not R_rel.shape == (3, 3):
        raise ValueError("Rotation matrices R1 and R2 must be 3x3.")
    if not t_rel.shape == (3, 1):
        raise ValueError("Translation vectors t1 and t2 must be 3x1.")
    if not n.shape == (3, 1):
        raise ValueError("Normal vector n must be 3x1.")
    if not isinstance(d, (int, float)):
        raise ValueError("Distance d must be a scalar.")

    # Calculate the intermediate matrix
    term = np.dot(t_rel, n.T) / d

    # Calculate the homography matrix
    H = np.dot(K2, np.dot(np.linalg.inv(R_rel) - term, np.linalg.inv(K1)))

    return H


def compute_homography_from_calib(calib_a: CameraCalibration, calib_b: CameraCalibration,
                                  distance: float) -> np.ndarray:
    return compute_homography(
        calib_a.intrinsics.camera_matrix,
        calib_b.intrinsics.camera_matrix,
        calib_a.extrinsics.rotation,
        calib_a.extrinsics.translation.reshape(3, 1) * 1000,
        np.array([[0], [0], [1]]),
        distance
    )


def project_pixels_to_points(uv, calibration: CameraCalibration):
    i = calibration.intrinsics
    xy = [0, 0]
    xy[0] = (uv[0] - i.cx) / i.fx
    xy[1] = (uv[1] - i.cy) / i.fy
    return xy


def unproject_points_to_pixels(xy, calibration: CameraCalibration):
    i = calibration.intrinsics
    undistorted_uv = [0, 0]
    undistorted_uv[0] = xy[0] * i.fx + i.cx
    undistorted_uv[1] = xy[1] * i.fy + i.cy
    return undistorted_uv


class CameraTransform:
    def __init__(self, color_calibration: CameraCalibration, depth_calibration: CameraCalibration):
        self._color_calibration = color_calibration
        self._depth_calibration = depth_calibration

        # pre-calculate distortion mappings
        self._color_distortion_mapping = compute_distortion_mapping(self._color_calibration)
        self._color_inv_distortion_mapping = compute_inverse_distortion_mapping(self._color_calibration)

        self._depth_distortion_mapping = compute_distortion_mapping(self._depth_calibration)
        self._depth_inv_distortion_mapping = compute_inverse_distortion_mapping(self._depth_calibration)

    def transform_2d_color_to_depth(self, uv: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        # todo: implement epipolar line optimisation from k4a for more accurate result
        # uv on color: goal uv on depth
        # pinhole model -> depth
        # 2d color -> depth => min_depth, max_depth
        # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/src/transformation/transformation.c#L325-L326

        # todo: find out, what the reprojection error is here that is optimized?
        color_camera_points = cv2.undistortPointsIter(
            uv.reshape(-1, 1, 2),
            self._color_calibration.intrinsics.camera_matrix,
            self._color_calibration.intrinsics.distortion_coefficients,
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
        )

        homogeneous_points = cv2.convertPointsToHomogeneous(color_camera_points).reshape(-1, 3)

        # set depth value to actual depth (multiply all components with depth / 1000)
        homogeneous_points *= depth_values_in_mm / 1000

        # get rotation and translation
        rotation_matrix = np.linalg.inv(self._color_calibration.extrinsics.rotation).astype(np.float64)
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        translation_vector = (self._color_calibration.extrinsics.translation * -1000)

        distorted_transformed_points, _ = cv2.projectPoints(
            homogeneous_points.reshape(-1, 1, 3),
            rotation_vector, translation_vector,
            self._depth_calibration.intrinsics.camera_matrix,
            self._depth_calibration.intrinsics.distortion_coefficients
        )

        return distorted_transformed_points.reshape(-1, 2)

    def transform_2d_depth_to_color(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        depth_camera_points = cv2.undistortPointsIter(
            pixels.reshape(-1, 1, 2),
            self._depth_calibration.intrinsics.camera_matrix,
            self._depth_calibration.intrinsics.distortion_coefficients,
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
        )

        # find depth values
        depth_uvs_int = np.round(pixels).astype(np.int32)
        depth_values = np.array([depth_map[y, x] for x, y in depth_uvs_int]).reshape(-1, 1)

        homogeneous_points = cv2.convertPointsToHomogeneous(depth_camera_points).reshape(-1, 3)

        # set depth value to actual depth (multiply all components with depth / 1000)
        homogeneous_points *= depth_values / 1000

        # get rotation and translation
        rotation_matrix = self._color_calibration.extrinsics.rotation.astype(np.float64)
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        translation_vector = (self._color_calibration.extrinsics.translation * 1000)

        distorted_transformed_points, _ = cv2.projectPoints(
            homogeneous_points.reshape(-1, 1, 3),
            rotation_vector, translation_vector,
            self._color_calibration.intrinsics.camera_matrix,
            self._color_calibration.intrinsics.distortion_coefficients
        )

        return distorted_transformed_points.reshape(-1, 2)

    def transform_depth_to_3d(self, uv: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        uv_int = np.round(uv).astype(np.int32)
        depth_values = depth_map[uv_int[:, 1], uv_int[:, 0]].reshape(-1, 1)
        object_points = np.hstack((uv, depth_values)).astype(np.float32)
        points_3d, _ = cv2.projectPoints(object_points, np.eye(1), np.zeros(3),
                                         self._depth_calibration.intrinsics.camera_matrix,
                                         self._depth_calibration.intrinsics.distortion_coefficients)
        return points_3d

    def transform_color_to_3d(self, uv: np.ndarray) -> np.ndarray:
        raise NotImplemented()
        return self.transform_depth_to_3d(self.transform_2d_color_to_depth(uv))
