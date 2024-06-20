import argparse
from typing import List, Optional

import cv2
import numpy as np

from openk4a.calibration import CameraCalibration
from openk4a.capture import OpenK4ACapture
from openk4a.playback import OpenK4APlayback
from playground.CharucoDetectionHelper import CharucoDetectionHelper
from playground.translation_example import compute_distortion_mapping, compute_inverse_distortion_mapping, \
    DistortionMapping
from playground.utils import normalize_image, concat_images_horizontally, annotate_points, concat_images_vertically


def k4a_project(xy: List[float], calibration: CameraCalibration) -> Optional[List[float]]:
    fx = calibration.intrinsics.fx
    fy = calibration.intrinsics.fy
    cx = calibration.intrinsics.cx
    cy = calibration.intrinsics.cy
    k1 = calibration.intrinsics.k1
    k2 = calibration.intrinsics.k2
    k3 = calibration.intrinsics.k3
    k4 = calibration.intrinsics.k4
    k5 = calibration.intrinsics.k5
    k6 = calibration.intrinsics.k6
    p1 = calibration.intrinsics.p1
    p2 = calibration.intrinsics.p2
    codx = 0  # assuming center of distortion is (0, 0) for Brown Conrady model
    cody = 0
    max_radius_for_projection = calibration.metric_radius

    # Validation check for fx and fy
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Expect both fx and fy to be larger than 0, actual values are fx: {fx}, fy: {fy}.")

    xp = xy[0] - codx
    yp = xy[1] - cody

    xp2 = xp * xp
    yp2 = yp * yp
    xyp = xp * yp
    rs = xp2 + yp2
    if rs > max_radius_for_projection * max_radius_for_projection:
        return None

    rss = rs * rs
    rsc = rss * rs
    a = 1.0 + k1 * rs + k2 * rss + k3 * rsc
    b = 1.0 + k4 * rs + k5 * rss + k6 * rsc
    bi = 1.0 / b if b != 0.0 else 1.0
    d = a * bi

    xp_d = xp * d
    yp_d = yp * d

    rs_2xp2 = rs + 2.0 * xp2
    rs_2yp2 = rs + 2.0 * yp2

    xp_d += rs_2xp2 * p2 + 2.0 * xyp * p1
    yp_d += rs_2yp2 * p1 + 2.0 * xyp * p2

    xp_d_cx = xp_d + codx
    yp_d_cy = yp_d + cody

    uv = [0, 0]
    uv[0] = xp_d_cx * fx + cx
    uv[1] = yp_d_cy * fy + cy

    return uv


def k4a_unproject(uv: List[float], calibration: CameraCalibration) -> List[float]:
    fx = calibration.intrinsics.fx
    fy = calibration.intrinsics.fy
    cx = calibration.intrinsics.cx
    cy = calibration.intrinsics.cy
    k1 = calibration.intrinsics.k1
    k2 = calibration.intrinsics.k2
    k3 = calibration.intrinsics.k3
    k4 = calibration.intrinsics.k4
    k5 = calibration.intrinsics.k5
    k6 = calibration.intrinsics.k6
    p1 = calibration.intrinsics.p1
    p2 = calibration.intrinsics.p2
    codx = 0  # assuming center of distortion is (0, 0) for Brown Conrady model
    cody = 0

    # correction for radial distortion
    xp_d = (uv[0] - cx) / fx - codx
    yp_d = (uv[1] - cy) / fy - cody

    rs = xp_d * xp_d + yp_d * yp_d
    rss = rs * rs
    rsc = rss * rs
    a = 1.0 + k1 * rs + k2 * rss + k3 * rsc
    b = 1.0 + k4 * rs + k5 * rss + k6 * rsc
    ai = 1.0 / a if a != 0.0 else 1.0
    di = ai * b

    xy = [0, 0]
    xy[0] = xp_d * di
    xy[1] = yp_d * di

    # approximate correction for tangential params
    two_xy = 2.0 * xy[0] * xy[1]
    xx = xy[0] * xy[0]
    yy = xy[1] * xy[1]

    xy[0] -= (yy + 3.0 * xx) * p2 + two_xy * p1
    xy[1] -= (xx + 3.0 * yy) * p1 + two_xy * p2

    # add on center of distortion
    xy[0] += codx
    xy[1] += cody

    # todo: call k4a_iterative_unproject

    return xy


def k4a_convert_to_normalized(uv, calibration: CameraCalibration):
    i = calibration.intrinsics
    xy = [0, 0]
    xy[0] = (uv[0] - i.cx) / i.fx
    xy[1] = (uv[1] - i.cy) / i.fy
    return xy


def k4a_convert_to_pixels(xy, calibration: CameraCalibration):
    i = calibration.intrinsics
    undistorted_uv = [0, 0]
    undistorted_uv[0] = xy[0] * i.fx + i.cx
    undistorted_uv[1] = xy[1] * i.fy + i.cy
    return undistorted_uv


def transform_2d_to_2d(points: np.ndarray, calib_a: CameraCalibration, calib_b: CameraCalibration,
                       distance: float) -> np.ndarray:
    """
    Transform 2D points from the coordinate system of one camera to another using their calibration data.

    Args:
        points (np.ndarray): Array of 2D points (Nx2) in the coordinate system of calib_a.
        calib_a (CameraCalibration): Calibration data for the source camera.
        calib_b (CameraCalibration): Calibration data for the destination camera.

    Returns:
        np.ndarray: Transformed 2D points (Nx2) in the coordinate system of calib_b.
    """
    H = compute_homography_from_calib(calib_a, calib_b, distance)
    return apply_homography(H, points)


def apply_homography(H: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    """
    Apply a homography transformation to 2D points.

    Args:
        H (np.ndarray): Homography matrix (3x3).
        points_uv (np.ndarray): Array of 2D points (Nx2) to be transformed.

    Returns:
        np.ndarray: Transformed 2D points (Nx2).
    """
    # Convert point to homogeneous coordinates
    uv_homogeneous = cv2.convertPointsToHomogeneous(points_uv).squeeze()

    # Apply homography
    uv_transformed_homogeneous = uv_homogeneous @ H.T
    uv_transformed = uv_transformed_homogeneous[:, :2]

    return uv_transformed.reshape(-1, 2)


def compute_homography_from_calib(calib_a: CameraCalibration, calib_b: CameraCalibration,
                                  distance: float) -> np.ndarray:
    return calculate_homography_openai(
        calib_a.intrinsics.camera_matrix,
        calib_b.intrinsics.camera_matrix,
        calib_a.extrinsics.rotation,
        calib_a.extrinsics.translation.reshape(3, 1) * 1000,
        calib_b.extrinsics.rotation,
        calib_b.extrinsics.translation.reshape(3, 1) * 1000,
        np.array([[0], [0], [1]]),
        distance
    )

    return compute_homography(
        calib_a.intrinsics.camera_matrix,
        calib_a.extrinsics.rotation,
        calib_a.extrinsics.translation[0] * 1000,
        calib_b.intrinsics.camera_matrix,
        calib_b.extrinsics.rotation,
        calib_b.extrinsics.translation[0] * 1000,
        distance
    )


def compute_homography(K1: np.ndarray, R1: np.ndarray, t1: np.ndarray, K2: np.ndarray, R2: np.ndarray, t2: np.ndarray,
                       distance: float) -> np.ndarray:
    """
    Compute the homography matrix between two cameras given their intrinsic and extrinsic parameters.

    Args:
        K1 (np.ndarray): Intrinsic matrix of the first camera.
        R1 (np.ndarray): Rotation matrix of the first camera.
        t1 (np.ndarray): Translation vector of the first camera.
        K2 (np.ndarray): Intrinsic matrix of the second camera.
        R2 (np.ndarray): Rotation matrix of the second camera.
        t2 (np.ndarray): Translation vector of the second camera.

    Returns:
        np.ndarray: Homography matrix (3x3) that transforms points from the first camera to the second.
    """
    # Compute the relative rotation and translation
    R_relative = R2 @ R1
    t_relative = R2.T @ t1 - R2.T @ t2

    # R_relative = np.identity(3)
    # t_relative = np.zeros(3)

    # define normal
    n = np.array([[0, 0, 1]])

    # Compute the homography
    # Ho12 = K2 * ( R2 - t2 * nT / do ) * K1-1

    to_CW = np.linalg.inv(K1)
    # to_CW_RT = (R_relative - t_relative.reshape(1, 3)) @ to_CW
    to_CW_RT = (R_relative - (t_relative.reshape(3, 1) @ n) / distance) @ to_CW
    to_K2 = K2 @ to_CW_RT

    return to_K2


def calculate_homography_openai(K1, K2, R1, t1, R2, t2, n, d):
    """
    Calculate the homography matrix H given the intrinsic and extrinsic parameters of two cameras,
    accounting for the tilt between the cameras.

    Parameters:
    - K1: Intrinsic matrix of the first camera (3x3 numpy array)
    - K2: Intrinsic matrix of the second camera (3x3 numpy array)
    - R1: Rotation matrix of the first camera (3x3 numpy array)
    - t1: Translation vector of the first camera (3x1 numpy array)
    - R2: Rotation matrix of the second camera (3x3 numpy array)
    - t2: Translation vector of the second camera (3x1 numpy array)
    - n: Normal vector to the plane (3x1 numpy array)
    - d: Distance from the origin to the plane along the normal (scalar)

    Returns:
    - H: Homography matrix (3x3 numpy array)
    """
    # Check if the input parameters are valid
    if not (K1.shape == (3, 3) and K2.shape == (3, 3)):
        raise ValueError("Intrinsic matrices K1 and K2 must be 3x3.")
    if not (R1.shape == (3, 3) and R2.shape == (3, 3)):
        raise ValueError("Rotation matrices R1 and R2 must be 3x3.")
    if not (t1.shape == (3, 1) and t2.shape == (3, 1)):
        raise ValueError("Translation vectors t1 and t2 must be 3x1.")
    if not (n.shape == (3, 1)):
        raise ValueError("Normal vector n must be 3x1.")
    if not isinstance(d, (int, float)):
        raise ValueError("Distance d must be a scalar.")

    # Calculate the intermediate matrix
    term = np.dot(t1, n.T) / d

    # Calculate the homography matrix
    H = np.dot(K2, np.dot(np.linalg.inv(R1) - term, np.linalg.inv(K1)))

    return H


def compute_homography_from_calib2(calib_a: CameraCalibration, calib_b: CameraCalibration, depth: float) -> np.ndarray:
    # Extract intrinsic parameters
    K_a = calib_a.intrinsics.camera_matrix
    K_b = calib_b.intrinsics.camera_matrix

    # Extract extrinsic parameters
    R_a = calib_a.extrinsics.rotation
    t_a = calib_a.extrinsics.translation.reshape((3, 1))

    R_b = calib_b.extrinsics.rotation
    t_b = calib_b.extrinsics.translation.reshape((3, 1))

    # Compute the relative rotation and translation
    R_rel = R_b @ np.linalg.inv(R_a)
    t_rel = t_b - R_rel @ t_a

    # Homography calculation with depth value for Z
    Rt = R_rel - (t_rel @ np.array([[0, 0, 1]], dtype=np.float32)) / depth
    H = K_b @ Rt @ np.linalg.inv(K_a)

    return H


def analyze_distortion(points: np.ndarray,
                       color: np.ndarray,
                       color_rectified: np.ndarray,
                       inv_color_distortion_mapping: DistortionMapping):
    annotate_points(color, points)

    points_rectified = inv_color_distortion_mapping.transform(points)
    annotate_points(color_rectified, points_rectified)

    cv2.imshow("Distortion Test", concat_images_vertically(color, color_rectified))
    cv2.waitKey(0)


def analyze_capture(azure: OpenK4APlayback, capture: OpenK4ACapture):
    detector = CharucoDetectionHelper()

    # prepare initial information
    color = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
    infrared = cv2.cvtColor(normalize_image(capture.ir), cv2.COLOR_GRAY2BGR)

    # prepare distortion mappings
    color_distortion_mapping = compute_distortion_mapping(azure.color_calibration)
    inv_color_distortion_mapping = compute_inverse_distortion_mapping(azure.color_calibration)
    infrared_distortion_mapping = compute_distortion_mapping(azure.depth_calibration)
    inv_infrared_distortion_mapping = compute_inverse_distortion_mapping(azure.depth_calibration)

    # distortion test
    point = np.array([[100, 100]], dtype=np.float32)
    k4a_camera_space = k4a_unproject(point[0], azure.color_calibration)
    k4a_uv_undist = k4a_convert_to_pixels(k4a_camera_space, azure.color_calibration)
    k4a_uv = k4a_project(k4a_camera_space, azure.color_calibration)
    opencv_inv_distortion = inv_color_distortion_mapping.transform(point)[0]
    opencv_distortion = color_distortion_mapping.transform(point)[0]

    # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/983
    size = (azure.color_calibration.width, azure.color_calibration.height)
    newK, _ = cv2.getOptimalNewCameraMatrix(azure.color_calibration.intrinsics.camera_matrix,
                                            azure.color_calibration.intrinsics.distortion_coefficients,
                                            size,
                                            1)

    map1, map2 = cv2.initInverseRectificationMap(newK, azure.color_calibration.intrinsics.distortion_coefficients,
                                                 np.eye(3), newK, size, cv2.CV_32FC1)
    optimal_inv_distortion = DistortionMapping(map1, map2)
    optimal_inv_camera_space = optimal_inv_distortion.transform(point)[0]
    optimal_color = cv2.remap(color, map1, map2, cv2.INTER_NEAREST)

    color_markers = detector.detect_markers(color)
    infrared_markers = detector.detect_markers(infrared)
    # markers.annotate(color)

    # extract only the origin point of each marker
    color_points = color_markers.corners[:, :, 0].reshape(-1, 2)
    infrared_points = infrared_markers.corners[:, :, 0].reshape(-1, 2)

    color_rectified = color_distortion_mapping.remap(color)
    infrared_rectified = infrared_distortion_mapping.remap(infrared)

    cv2.imshow("Optimal Matrix", concat_images_vertically(color_rectified, optimal_color))

    center_distance = capture.depth[576 // 2, 640 // 2]

    # experiment correct distortion mapping
    # analyze_distortion(color_points, color.copy(), color_rectified.copy(), inv_color_distortion_mapping)
    # analyze_distortion(infrared_points, infrared.copy(), infrared_rectified.copy(), inv_infrared_distortion_mapping)

    color_points_rectified = inv_color_distortion_mapping.transform(color_points)

    # compute homography from calibration
    H = compute_homography_from_calib(azure.color_calibration, azure.depth_calibration, float(center_distance) / 1000)
    ir_points_rectified = cv2.perspectiveTransform(color_points_rectified.reshape(-1, 1, 2), H).reshape(-1, 2)

    # ir_points_rectified = transform_points_with_cv(color_points_rectified, azure.color_calibration, azure.depth_calibration, 0.8)

    # ir_points_rectified[:, 0] = 640 - ir_points_rectified[:, 0]
    # ir_points_rectified[:, 1] = 576 - ir_points_rectified[:, 1]

    # warp test
    h, w = infrared_rectified.shape[:2]
    color_warped = cv2.warpPerspective(color_rectified, H, (w, h))
    annotate_points(color_warped, ir_points_rectified)
    cv2.imshow("Warped", concat_images_horizontally(color_warped, infrared_rectified))

    annotate_points(color_rectified, color_points_rectified)
    annotate_points(infrared_rectified, ir_points_rectified)

    cv2.imshow("Result IR", concat_images_horizontally(color_rectified, infrared_rectified))
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = True
    azure.open()

    while capture := azure.read():
        analyze_capture(azure, capture)

    azure.close()


if __name__ == "__main__":
    main()
