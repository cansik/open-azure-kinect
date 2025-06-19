from dataclasses import dataclass

import cv2
import numpy as np

from openk4a.calibration import CameraCalibration


@dataclass
class DistortionMapping:
    x_mapping: np.ndarray  # shape (H, W)
    y_mapping: np.ndarray  # shape (H, W)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized lookup of the undistorted coordinates for a list of pixels.

        points: (N,2) float or int array of (u, v) pixel indices
        returns: (N,2) float array of (undist_x, undist_y)
        """
        uv = points.astype(np.int32)
        u = uv[:, 0]
        v = uv[:, 1]

        undist_x = self.x_mapping[v, u]
        undist_y = self.y_mapping[v, u]
        return np.stack([undist_x, undist_y], axis=1)

    def remap(self, image: np.ndarray, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
        """
        Apply the precomputed undistort maps to the whole image in one call.
        """
        return cv2.remap(image, self.x_mapping, self.y_mapping, interpolation)


def compute_distortion_mapping(calibration: CameraCalibration) -> DistortionMapping:
    """
    Calculates the mapping from undistorted image space to distorted image space.
    :param calibration:
    :return:
    """
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
    """
    Calculates the mapping from distorted image space to undistorted image space.
    :param calibration:
    :return:
    """
    x_map, y_map = cv2.initInverseRectificationMap(
        calibration.intrinsics.camera_matrix,
        calibration.intrinsics.distortion_coefficients,
        np.eye(3),
        calibration.intrinsics.camera_matrix,
        (calibration.width, calibration.height),
        5,  # CV_32FC1
    )

    return DistortionMapping(x_map, y_map)


def compute_inverse_distortion_mapping_exact(calibration: CameraCalibration) -> DistortionMapping:
    """
    Calculates the mapping from distorted image space to undistorted image space
    by directly solving the distortion equations at every pixel via undistortPointsIter.
    This yields sub‐pixel‐exact maps at the cost of one heavy init pass.
    """
    K = calibration.intrinsics.camera_matrix
    dist = calibration.intrinsics.distortion_coefficients
    fx, fy = calibration.intrinsics.fx, calibration.intrinsics.fy
    cx, cy = calibration.intrinsics.cx, calibration.intrinsics.cy
    W, H = calibration.width, calibration.height

    # build a full grid of distorted pixel coords [0..W)×[0..H)
    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    grid_uv = np.stack(np.meshgrid(us, vs), axis=-1)  # shape (H, W, 2)
    pts = grid_uv.reshape(-1, 1, 2)  # shape (H*W,1,2)

    # undistortPointsIter to normalized coords
    norm_pts = cv2.undistortPointsIter(
        pts, K, dist,
        None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
    )  # shape (H*W,1,2)

    # reproject to pixel space
    norm_pts = norm_pts.reshape(-1, 2)
    map_x_flat = norm_pts[:, 0] * fx + cx
    map_y_flat = norm_pts[:, 1] * fy + cy

    # reshape back into H×W float32 maps
    map_x = map_x_flat.reshape(H, W).astype(np.float32)
    map_y = map_y_flat.reshape(H, W).astype(np.float32)

    return DistortionMapping(map_x, map_y)


class CameraTransform:
    def __init__(self, color_calibration: CameraCalibration, depth_calibration: CameraCalibration):
        self._color_calibration = color_calibration
        self._depth_calibration = depth_calibration

        # pre-calculate distortion mappings
        # for the inverse distortion mapping we use the exact (optimised) method
        self._color_distortion_mapping = compute_distortion_mapping(self._color_calibration)
        self._color_inv_distortion_mapping = compute_inverse_distortion_mapping_exact(self._color_calibration)

        self._depth_distortion_mapping = compute_distortion_mapping(self._depth_calibration)
        self._depth_inv_distortion_mapping = compute_inverse_distortion_mapping_exact(self._depth_calibration)

    def transform_2d_depth_to_color(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        norm = self._pixels_to_normalized_plane(
            pixels, self._depth_calibration, self._depth_inv_distortion_mapping
        )
        Z = self._pixels_to_depth(pixels, depth_map) / 1000
        pts_depth = np.stack([norm[:, 0] * Z, norm[:, 1] * Z, Z], axis=1)

        # Transform points from depth frame into color frame
        rot_vec, trans_vec = self._get_relative_extrinsics(
            self._depth_calibration, self._color_calibration
        )
        projected, _ = cv2.projectPoints(
            pts_depth.reshape(-1, 1, 3),
            rot_vec, trans_vec,
            self._color_calibration.intrinsics.camera_matrix,
            self._color_calibration.intrinsics.distortion_coefficients
        )
        return projected.reshape(-1, 2)

    def transform_2d_color_to_depth(self, pixels: np.ndarray, depth_map: np.ndarray,
                                    z_near: float = 0.2, z_far: float = 5.0,
                                    steps: int = 50) -> np.ndarray:
        # Undistort and normalize color image pixels
        color_norm = self._pixels_to_normalized_plane(
            pixels, self._color_calibration, self._color_inv_distortion_mapping
        )

        # Perform epipolar depth search
        depths = self._epipolar_search(color_norm, depth_map, z_near, z_far, steps)

        # Back-project to 3D in color camera frame
        pts_color_cam = np.hstack([color_norm, np.ones((color_norm.shape[0], 1))])
        pts_color_cam *= depths.reshape(-1, 1)

        # Compute relative transform from color to depth
        rot_vec, trans_vec = self._get_relative_extrinsics(
            self._color_calibration, self._depth_calibration
        )

        # Project into depth image
        projected, _ = cv2.projectPoints(
            pts_color_cam.reshape(-1, 1, 3),
            rot_vec, trans_vec,
            self._depth_calibration.intrinsics.camera_matrix,
            self._depth_calibration.intrinsics.distortion_coefficients
        )
        return projected.reshape(-1, 2)

    def _epipolar_search(self, color_norm: np.ndarray, depth_map: np.ndarray,
                         z_near: float, z_far: float, steps: int) -> np.ndarray:
        H, W = depth_map.shape
        N = color_norm.shape[0]
        depths = np.zeros(N, dtype=np.float32)
        sample_z = np.linspace(z_near, z_far, steps, dtype=np.float32)

        # Precompute relative transform
        rot_vec, trans_vec = self._get_relative_extrinsics(
            self._color_calibration, self._depth_calibration
        )

        for i in range(N):
            ray = np.tile(color_norm[i], (steps, 1))
            pts = np.hstack([ray, np.ones((steps, 1), dtype=np.float32)])
            pts *= sample_z.reshape(-1, 1)

            proj_pts, _ = cv2.projectPoints(
                pts.reshape(-1, 1, 3), rot_vec, trans_vec,
                self._depth_calibration.intrinsics.camera_matrix,
                self._depth_calibration.intrinsics.distortion_coefficients
            )
            proj = proj_pts.reshape(-1, 2)

            u = np.clip(np.round(proj[:, 0]).astype(int), 0, W - 1)
            v = np.clip(np.round(proj[:, 1]).astype(int), 0, H - 1)
            depth_vals = depth_map[v, u] / 1000.0

            diff = np.abs(depth_vals - sample_z)
            best_idx = np.argmin(diff)
            depths[i] = sample_z[best_idx]

        return depths

    def transform_depth_to_3d(
            self,
            pixels: np.ndarray,
            depth_map: np.ndarray
    ) -> np.ndarray:
        norm = self._pixels_to_normalized_plane(
            pixels, self._depth_calibration, self._depth_inv_distortion_mapping
        )

        # lookup depth with interpolation in m
        Z = self._pixels_to_depth(pixels, depth_map) / 1000

        # back-project: X_mm = x_norm * Z_mm, etc.
        pts = np.empty((pixels.shape[0], 3), dtype=np.float32)
        pts[:, 0] = norm[:, 0] * Z
        pts[:, 1] = norm[:, 1] * Z
        pts[:, 2] = Z

        return pts

    def transform_color_to_3d(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        return self.transform_depth_to_3d(self.transform_2d_color_to_depth(pixels, depth_map), depth_map)

    def create_pointcloud(self, depth_map: np.ndarray, stride: int = 1) -> np.ndarray:
        height, width = depth_map.shape

        u_vals = np.arange(0, width, stride, dtype=np.float32)
        v_vals = np.arange(0, height, stride, dtype=np.float32)
        uu, vv = np.meshgrid(u_vals, v_vals)  # shapes (H/stride, W/stride)
        sampled_pixels = np.stack([uu, vv], axis=-1).reshape(-1, 2)

        return self.transform_depth_to_3d(sampled_pixels, depth_map)

    @staticmethod
    def _pixels_to_normalized_plane(pixels: np.ndarray,
                                    calibration: CameraCalibration,
                                    inv_map: DistortionMapping) -> np.ndarray:
        """
        Undistort pixel coordinates and convert to normalized image-plane coords (x_norm, y_norm).
        """
        # Undistort using precomputed inverse map
        undistorted = inv_map.transform(pixels.astype(np.int32, copy=False))
        # Fetch intrinsics
        fx = calibration.intrinsics.fx
        fy = calibration.intrinsics.fy
        cx = calibration.intrinsics.cx
        cy = calibration.intrinsics.cy
        # Convert to normalized coordinates
        x_norm = (undistorted[:, 0] - cx) / fx
        y_norm = (undistorted[:, 1] - cy) / fy
        return np.stack([x_norm, y_norm], axis=1)

    @staticmethod
    def _pixels_to_depth(pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Lookup the nearest depth (in millimeters) for each pixel coordinate
        by rounding to the nearest integer pixel.
        """
        H, W = depth_map.shape
        uv = np.rint(pixels).astype(np.int32)
        u = np.clip(uv[:, 0], 0, W - 1)
        v = np.clip(uv[:, 1], 0, H - 1)
        return depth_map[v, u].astype(np.float32)

    @staticmethod
    def _get_relative_extrinsics(src_cal, dst_cal):
        """
        Compute rotation and translation to go from src camera frame to dst camera frame.
        Returns:
            rot_vec: Rodrigues rotation vector
            trans_vec: translation vector (3,)
        """
        # World transformations: X_world = R_src * X_src + t_src
        R_src = src_cal.extrinsics.rotation.astype(np.float64)
        t_src = src_cal.extrinsics.translation.astype(np.float64).reshape(3, 1)
        R_dst = dst_cal.extrinsics.rotation.astype(np.float64)
        t_dst = dst_cal.extrinsics.translation.astype(np.float64).reshape(3, 1)

        # Relative rotation: R_rel = R_dst * R_src.T
        R_rel = R_dst.dot(R_src.T)
        # Relative translation: t_rel = t_dst - R_rel * t_src
        t_rel = t_dst - R_rel.dot(t_src)

        rot_vec, _ = cv2.Rodrigues(R_rel)
        return rot_vec, t_rel.flatten()
