import cv2
import numpy as np

from openk4a.calibration import CameraCalibration


class CameraTransform:
    def __init__(self, color_calibration: CameraCalibration, depth_calibration: CameraCalibration):
        self._color_calibration = color_calibration
        self._depth_calibration = depth_calibration

    def transform_2d_depth_to_color(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        # Undistort and normalize depth image pixels
        depth_norm = cv2.undistortPointsIter(
            pixels.reshape(-1, 1, 2),
            self._depth_calibration.intrinsics.camera_matrix,
            self._depth_calibration.intrinsics.distortion_coefficients,
            None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
        ).reshape(-1, 2)

        # Remap to get depth values and back-project to 3D
        map_x = pixels[:, 0].reshape(-1, 1).astype(np.float32)
        map_y = pixels[:, 1].reshape(-1, 1).astype(np.float32)
        depth_values = cv2.remap(depth_map, map_x, map_y, interpolation=cv2.INTER_LINEAR).reshape(-1, 1)
        pts_depth_cam = np.hstack([depth_norm, np.ones((depth_norm.shape[0], 1))])
        pts_depth_cam *= (depth_values / 1000)

        # Compute relative transform from depth to color
        rot_vec, trans_vec = self._get_relative_extrinsics(
            self._depth_calibration, self._color_calibration
        )

        # Project into color image
        projected, _ = cv2.projectPoints(
            pts_depth_cam.reshape(-1, 1, 3),
            rot_vec, trans_vec,
            self._color_calibration.intrinsics.camera_matrix,
            self._color_calibration.intrinsics.distortion_coefficients
        )
        return projected.reshape(-1, 2)

    def transform_2d_color_to_depth(self, pixels: np.ndarray, depth_map: np.ndarray,
                                    z_near: float = 0.2, z_far: float = 5.0,
                                    steps: int = 50) -> np.ndarray:
        # Undistort and normalize color image pixels
        color_norm = cv2.undistortPointsIter(
            pixels.reshape(-1, 1, 2),
            self._color_calibration.intrinsics.camera_matrix,
            self._color_calibration.intrinsics.distortion_coefficients,
            None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
        ).reshape(-1, 2)

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

    def transform_depth_to_3d(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        uv_int = pixels.astype(np.int32)
        depth_values = depth_map[uv_int[:, 1], uv_int[:, 0]].reshape(-1, 1)
        object_points = np.hstack((pixels, depth_values)).astype(np.float32)
        points_3d, _ = cv2.projectPoints(object_points, np.eye(1), np.zeros(3),
                                         self._depth_calibration.intrinsics.camera_matrix,
                                         self._depth_calibration.intrinsics.distortion_coefficients)
        return points_3d

    def transform_color_to_3d(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        return self.transform_depth_to_3d(self.transform_2d_color_to_depth(pixels, depth_map), depth_map)
