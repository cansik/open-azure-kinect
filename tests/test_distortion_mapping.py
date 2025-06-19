from typing import List

import cv2
import numpy as np
import pytest

from openk4a.calibration import CameraCalibration
from openk4a.transform import compute_inverse_distortion_mapping, compute_distortion_mapping
from tests.test_utils import camera_calibrations

CALIBRATIONS: List[CameraCalibration] = camera_calibrations


@pytest.mark.parametrize("camera_calibration", CALIBRATIONS)
def test_forward_undistort_map_accuracy(camera_calibration: CameraCalibration) -> None:
    """Measure the max and mean pixel‐error between undistortPointsIter reprojection and precomputed map."""
    # Intrinsics
    fx: float = camera_calibration.intrinsics.fx
    fy: float = camera_calibration.intrinsics.fy
    cx: float = camera_calibration.intrinsics.cx
    cy: float = camera_calibration.intrinsics.cy
    dist_coeffs: np.ndarray = camera_calibration.intrinsics.distortion_coefficients
    K: np.ndarray = camera_calibration.intrinsics.camera_matrix

    # Grid of distorted pixels
    cols, rows = 20, 20
    u_vals = np.linspace(0, camera_calibration.width - 1, cols, dtype=int)
    v_vals = np.linspace(0, camera_calibration.height - 1, rows, dtype=int)
    distorted = np.stack(np.meshgrid(u_vals, v_vals), axis=-1).reshape(-1, 2).astype(np.float32)

    # Reference via undistortPointsIter + reprojection
    undist_norm = cv2.undistortPointsIter(
        distorted.reshape(-1, 1, 2), K, dist_coeffs,
        None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
    ).reshape(-1, 2)
    expected = np.empty_like(undist_norm)
    expected[:, 0] = undist_norm[:, 0] * fx + cx
    expected[:, 1] = undist_norm[:, 1] * fy + cy

    # Precomputed inverse‐map (undistortion)
    inv_map = compute_inverse_distortion_mapping(camera_calibration)
    got = inv_map.transform(distorted)

    # Compute error metrics
    diffs = np.abs(expected - got)
    max_error = float(diffs.max())
    mean_error = float(diffs.mean())

    # Print a single accuracy summary
    print(f"Camera ({camera_calibration.width}×{camera_calibration.height}): "
          f"max error = {max_error:.6f} px, mean error = {mean_error:.6f} px")

    # Assert that the mean_error error is small (0.5px mean)
    tol = 0.5
    assert mean_error <= tol, (
        f"Mean error {max_error:.6f} px exceeds tolerance {tol} px"
    )


@pytest.mark.parametrize("camera_calibration", CALIBRATIONS)
def test_inverse_distortion_map_accuracy(camera_calibration: CameraCalibration) -> None:
    """
    Measure the max and mean pixel‐error between cv2.projectPoints distortion
    and the precomputed forward DistortionMapping.
    """
    # Intrinsics
    fx: float = camera_calibration.intrinsics.fx
    fy: float = camera_calibration.intrinsics.fy
    cx: float = camera_calibration.intrinsics.cx
    cy: float = camera_calibration.intrinsics.cy
    dist_coeffs: np.ndarray = camera_calibration.intrinsics.distortion_coefficients
    K: np.ndarray = camera_calibration.intrinsics.camera_matrix

    # Grid of undistorted pixels
    cols, rows = 20, 20
    u_vals = np.linspace(0, camera_calibration.width - 1, cols, dtype=int)
    v_vals = np.linspace(0, camera_calibration.height - 1, rows, dtype=int)
    undistorted = (
        np.stack(np.meshgrid(u_vals, v_vals), axis=-1)
        .reshape(-1, 2)
        .astype(np.float32)
    )

    # Convert to normalized coords (Z=1)
    norm = np.empty_like(undistorted)
    norm[:, 0] = (undistorted[:, 0] - cx) / fx
    norm[:, 1] = (undistorted[:, 1] - cy) / fy
    pts3d = np.hstack([norm, np.ones((norm.shape[0], 1), dtype=np.float32)])  # (N,3)

    # Reference distortion via projectPoints
    projected, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3),
        rvec=np.zeros(3, dtype=np.float32),
        tvec=np.zeros(3, dtype=np.float32),
        cameraMatrix=K,
        distCoeffs=dist_coeffs
    )
    expected = projected.reshape(-1, 2)  # (N,2)

    # Precomputed forward‐distortion map
    fwd_map = compute_distortion_mapping(camera_calibration)
    got = fwd_map.transform(undistorted)

    # Compute error metrics
    diffs = np.abs(expected - got)
    errors = np.linalg.norm(diffs, axis=1)
    max_error = float(errors.max())
    mean_error = float(errors.mean())

    # Print summary
    print(
        f"Camera ({camera_calibration.width}×{camera_calibration.height}): "
        f"max error = {max_error:.6f} px, mean error = {mean_error:.6f} px"
    )

    # Assert that the mean error is small (e.g. 0.5 px)
    tol = 0.5
    assert mean_error <= tol, (
        f"Mean error {mean_error:.6f} px exceeds tolerance {tol} px"
    )
