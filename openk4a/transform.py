from dataclasses import dataclass

import cv2
import numpy as np

from openk4a.calibration import CameraCalibration


@dataclass
class DistortionMapping:
    """
    Data class representing a distortion mapping with pixel-wise x and y coordinate maps.

    :param x_mapping: Array of shape (H, W) representing x-coordinates for remapping.
    :param y_mapping: Array of shape (H, W) representing y-coordinates for remapping.
    """
    x_mapping: np.ndarray
    y_mapping: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized lookup of the undistorted coordinates for a list of pixels.

        :param points: Array of shape (N, 2) with float or int pixel indices (u, v).

        :returns: Array of shape (N, 2) with float undistorted coordinates (x, y).
        """
        uv = points.astype(np.int32)
        u = uv[:, 0]
        v = uv[:, 1]

        undist_x = self.x_mapping[v, u]
        undist_y = self.y_mapping[v, u]
        return np.stack([undist_x, undist_y], axis=1)

    def remap(self, image: np.ndarray, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
        """
        Applies the precomputed undistortion maps to the input image.

        :param image: Input image to be remapped.
        :param interpolation: OpenCV interpolation method, default is INTER_NEAREST.

        :returns: Undistorted image.
        """
        return cv2.remap(image, self.x_mapping, self.y_mapping, interpolation)


def compute_distortion_mapping(calibration: CameraCalibration) -> DistortionMapping:
    """
    Computes the distortion mapping from undistorted image space to distorted image space.

    :param calibration: CameraCalibration object containing intrinsics and image size.

    :returns: DistortionMapping object with remap coordinates.
    """
    x_mapping, y_mapping = cv2.initUndistortRectifyMap(
        calibration.intrinsics.camera_matrix,
        calibration.intrinsics.distortion_coefficients,
        np.eye(3),
        calibration.intrinsics.camera_matrix,
        (calibration.width, calibration.height),
        5  # CV_32FC1
    )
    return DistortionMapping(x_mapping, y_mapping)


def compute_inverse_distortion_mapping(calibration: CameraCalibration) -> DistortionMapping:
    """
    Computes the inverse distortion mapping from distorted image space to undistorted image space.

    :param calibration: CameraCalibration object containing intrinsics and image size.

    :returns: DistortionMapping object with inverse remap coordinates.
    """
    x_mapping, y_mapping = cv2.initInverseRectificationMap(
        calibration.intrinsics.camera_matrix,
        calibration.intrinsics.distortion_coefficients,
        np.eye(3),
        calibration.intrinsics.camera_matrix,
        (calibration.width, calibration.height),
        5  # CV_32FC1
    )
    return DistortionMapping(x_mapping, y_mapping)


def compute_inverse_distortion_mapping_exact(calibration: CameraCalibration) -> DistortionMapping:
    """
    Computes a subpixel-accurate inverse distortion mapping using iterative undistortion.

    This method uses cv2.undistortPointsIter to solve for normalized coordinates at each pixel.

    :param calibration: CameraCalibration object containing intrinsics and image size.

    :returns: DistortionMapping object with high-precision inverse remap coordinates.
    """
    camera_matrix = calibration.intrinsics.camera_matrix
    distortion_coeffs = calibration.intrinsics.distortion_coefficients
    focal_x, focal_y = calibration.intrinsics.fx, calibration.intrinsics.fy
    center_x, center_y = calibration.intrinsics.cx, calibration.intrinsics.cy
    width, height = calibration.width, calibration.height

    u_values = np.arange(width, dtype=np.float32)
    v_values = np.arange(height, dtype=np.float32)
    grid_uv = np.stack(np.meshgrid(u_values, v_values), axis=-1)
    pixel_points = grid_uv.reshape(-1, 1, 2)

    normalized_points = cv2.undistortPointsIter(
        pixel_points, camera_matrix, distortion_coeffs,
        None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-22)
    )
    normalized_points = normalized_points.reshape(-1, 2)
    map_x = (normalized_points[:, 0] * focal_x + center_x).reshape(height, width).astype(np.float32)
    map_y = (normalized_points[:, 1] * focal_y + center_y).reshape(height, width).astype(np.float32)

    return DistortionMapping(map_x, map_y)


class CameraTransform:
    """
    A class that encapsulates camera calibration and transformations between depth and color spaces.
    """

    def __init__(self, color_calibration: CameraCalibration, depth_calibration: CameraCalibration):
        """
        Initializes the camera transformation class with color and depth camera calibration.

        :param color_calibration: Calibration for the color camera.
        :param depth_calibration: Calibration for the depth camera.
        """
        self._color_calibration = color_calibration
        self._depth_calibration = depth_calibration

        self._color_distortion_mapping = compute_distortion_mapping(self._color_calibration)
        self._color_inv_distortion_mapping = compute_inverse_distortion_mapping_exact(self._color_calibration)

        self._depth_distortion_mapping = compute_distortion_mapping(self._depth_calibration)
        self._depth_inv_distortion_mapping = compute_inverse_distortion_mapping_exact(self._depth_calibration)

    def transform_2d_depth_to_color(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Projects 2D pixels from the depth image to the color image coordinate space.

        :param pixels: (N, 2) array of depth image pixels.
        :param depth_map: Depth map array.

        :returns: (N, 2) array of corresponding coordinates in the color image.
        """
        normalized = self._pixels_to_normalized_plane(pixels, self._depth_calibration,
                                                      self._depth_inv_distortion_mapping)
        depth_values = self._pixels_to_depth(pixels, depth_map) / 1000
        points_depth = np.stack([normalized[:, 0] * depth_values, normalized[:, 1] * depth_values, depth_values],
                                axis=1)

        rotation_vector, translation_vector = self._get_relative_extrinsics(self._depth_calibration,
                                                                            self._color_calibration)
        projected, _ = cv2.projectPoints(
            points_depth.reshape(-1, 1, 3),
            rotation_vector, translation_vector,
            self._color_calibration.intrinsics.camera_matrix,
            self._color_calibration.intrinsics.distortion_coefficients
        )
        return projected.reshape(-1, 2)

    def transform_2d_color_to_depth(self, pixels: np.ndarray, depth_map: np.ndarray,
                                    z_near: float = 0.2, z_far: float = 5.0,
                                    steps: int = 50) -> np.ndarray:
        """
        Projects 2D pixels from the color image to the depth image coordinate space via epipolar search.

        :param pixels: (N, 2) array of color image pixels.
        :param depth_map: Depth map array.
        :param z_near: Near clipping distance for epipolar search in meters.
        :param z_far: Far clipping distance for epipolar search in meters.
        :param steps: Number of steps to evaluate along the epipolar line.

        :returns: (N, 2) array of corresponding coordinates in the depth image.
        """
        normalized_color = self._pixels_to_normalized_plane(pixels, self._color_calibration,
                                                            self._color_inv_distortion_mapping)
        depths = self._epipolar_search(normalized_color, depth_map, z_near, z_far, steps)

        points_color_camera = np.hstack([normalized_color, np.ones((normalized_color.shape[0], 1))])
        points_color_camera *= depths.reshape(-1, 1)

        rotation_vector, translation_vector = self._get_relative_extrinsics(self._color_calibration,
                                                                            self._depth_calibration)

        projected, _ = cv2.projectPoints(
            points_color_camera.reshape(-1, 1, 3),
            rotation_vector, translation_vector,
            self._depth_calibration.intrinsics.camera_matrix,
            self._depth_calibration.intrinsics.distortion_coefficients
        )
        return projected.reshape(-1, 2)

    def transform_depth_to_3d(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Back-projects 2D depth pixels into 3D space in the depth camera coordinate system.

        :param pixels: (N, 2) array of pixel coordinates.
        :param depth_map: Depth map array.

        :returns: (N, 3) array of 3D points in camera space.
        """
        normalized = self._pixels_to_normalized_plane(pixels, self._depth_calibration,
                                                      self._depth_inv_distortion_mapping)
        depth_values = self._pixels_to_depth(pixels, depth_map) / 1000

        points_3d = np.empty((pixels.shape[0], 3), dtype=np.float32)
        points_3d[:, 0] = normalized[:, 0] * depth_values
        points_3d[:, 1] = normalized[:, 1] * depth_values
        points_3d[:, 2] = depth_values
        return points_3d

    def transform_color_to_3d(self, pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Projects 2D color image pixels into 3D space using aligned depth.

        :param pixels: (N, 2) array of color image coordinates.
        :param depth_map: Depth map array.

        :returns: (N, 3) array of 3D points in color camera space.
        """
        depth_pixels = self.transform_2d_color_to_depth(pixels, depth_map)
        return self.transform_depth_to_3d(depth_pixels, depth_map)

    def create_pointcloud(self, depth_map: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Generates a 3D point cloud from the depth image.

        :param depth_map: Input depth map.
        :param stride: Sampling stride to reduce density.

        :returns: (M, 3) array of 3D points.
        """
        height, width = depth_map.shape
        sampled_pixels = self._create_pixel_grid(width, height, stride)

        return self.transform_depth_to_3d(sampled_pixels, depth_map)

    def align_image_depth_to_color(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Efficiently aligns an image from the depth camera space (e.g., infrared)
        into the color camera coordinate frame using forward warping.

        :param image: Image in depth camera space (e.g., 640x576 IR).
        :param depth_map: Depth map of same resolution as `image`.

        :returns: Aligned image in color camera space (e.g., 1920x1080).
        """
        height, width = image.shape
        pixel_grid = self._create_pixel_grid(width, height)

        # Transform to color camera space
        mapped_pixels = self.transform_2d_depth_to_color(pixel_grid, depth_map)

        # Prepare output canvas
        out_height, out_width = self._color_calibration.height, self._color_calibration.width
        output = np.zeros((out_height, out_width), dtype=image.dtype)

        # Round to integer pixel coordinates
        x = np.rint(mapped_pixels[:, 0]).astype(int)
        y = np.rint(mapped_pixels[:, 1]).astype(int)

        # Clamp to valid range
        valid = (x >= 0) & (x < out_width) & (y >= 0) & (y < out_height)
        x = x[valid]
        y = y[valid]
        values = image.ravel()[valid]

        output[y, x] = values  # Forward warp with nearest neighbor

        return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else output

    def align_image_color_to_depth(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Aligns a color image to the depth camera coordinate frame.

        :param image: Color image in the color camera coordinate space.
        :param depth_map: Reference depth map for lookup and interpolation.

        :returns: Color image remapped to the depth camera space.
        """
        height, width = depth_map.shape
        pixel_grid = self._create_pixel_grid(width, height)
        mapped_pixels = self.transform_2d_depth_to_color(pixel_grid, depth_map)

        map_x = mapped_pixels[:, 0].reshape(height, width).astype(np.float32)
        map_y = mapped_pixels[:, 1].reshape(height, width).astype(np.float32)

        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def _epipolar_search(self, color_norm: np.ndarray, depth_map: np.ndarray,
                         z_near: float, z_far: float, steps: int) -> np.ndarray:
        """
        Searches along epipolar lines to find the most plausible depth match for each color pixel.

        :param color_norm: (N, 2) normalized color coordinates.
        :param depth_map: Input depth map.
        :param z_near: Near depth bound in meters.
        :param z_far: Far depth bound in meters.
        :param steps: Number of search steps.

        :returns: (N,) array of matched depth values.
        """
        height, width = depth_map.shape
        num_points = color_norm.shape[0]
        sample_depths = np.linspace(z_near, z_far, steps, dtype=np.float32)

        xx = (color_norm[:, 0][:, None] * sample_depths[None, :]).reshape(-1)
        yy = (color_norm[:, 1][:, None] * sample_depths[None, :]).reshape(-1)
        zz = np.tile(sample_depths, num_points)
        points_color = np.stack([xx, yy, zz], axis=1).astype(np.float32).reshape(-1, 1, 3)

        rotation_vector, translation_vector = self._get_relative_extrinsics(self._color_calibration,
                                                                            self._depth_calibration)
        projection_matrix = self._depth_calibration.intrinsics.camera_matrix
        distortion_coeffs = self._depth_calibration.intrinsics.distortion_coefficients

        projected, _ = cv2.projectPoints(points_color, rotation_vector, translation_vector, projection_matrix,
                                         distortion_coeffs)
        projected = projected.reshape(num_points, steps, 2)

        u_indices = np.clip(np.round(projected[..., 0]).astype(int), 0, width - 1)
        v_indices = np.clip(np.round(projected[..., 1]).astype(int), 0, height - 1)
        sampled_depths = depth_map[v_indices, u_indices] / 1000.0

        differences = np.abs(sampled_depths - sample_depths[None, :])
        best_indices = np.argmin(differences, axis=1)
        return sample_depths[best_indices]

    @staticmethod
    def _pixels_to_normalized_plane(pixels: np.ndarray,
                                    calibration: CameraCalibration,
                                    inv_map: DistortionMapping) -> np.ndarray:
        """
        Converts distorted pixel coordinates to normalized image plane coordinates.

        :param pixels: (N, 2) array of distorted pixel coordinates.
        :param calibration: Camera calibration containing intrinsics.
        :param inv_map: Precomputed inverse distortion map.

        :returns: (N, 2) array of normalized coordinates.
        """
        undistorted = inv_map.transform(pixels.astype(np.int32, copy=False))
        fx = calibration.intrinsics.fx
        fy = calibration.intrinsics.fy
        cx = calibration.intrinsics.cx
        cy = calibration.intrinsics.cy

        x_norm = (undistorted[:, 0] - cx) / fx
        y_norm = (undistorted[:, 1] - cy) / fy
        return np.stack([x_norm, y_norm], axis=1)

    @staticmethod
    def _pixels_to_depth(pixels: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Retrieves the depth values in millimeters for each given pixel.

        :param pixels: (N, 2) array of pixel coordinates.
        :param depth_map: Input depth map.

        :returns: (N,) array of depth values in millimeters.
        """
        height, width = depth_map.shape
        uv = np.rint(pixels).astype(np.int32)
        u = np.clip(uv[:, 0], 0, width - 1)
        v = np.clip(uv[:, 1], 0, height - 1)
        return depth_map[v, u].astype(np.float32)

    @staticmethod
    def _get_relative_extrinsics(src_cal, dst_cal):
        """
        Computes the relative extrinsic transformation from the source to the destination camera frame.

        :param src_cal: Source camera calibration.
        :param dst_cal: Destination camera calibration.

        :returns: Tuple of (rotation vector, translation vector).
        """
        rotation_src = src_cal.extrinsics.rotation.astype(np.float64)
        translation_src = src_cal.extrinsics.translation.astype(np.float64).reshape(3, 1)
        rotation_dst = dst_cal.extrinsics.rotation.astype(np.float64)
        translation_dst = dst_cal.extrinsics.translation.astype(np.float64).reshape(3, 1)

        relative_rotation = rotation_dst.dot(rotation_src.T)
        relative_translation = translation_dst - relative_rotation.dot(translation_src)

        rotation_vector, _ = cv2.Rodrigues(relative_rotation)
        return rotation_vector, relative_translation.flatten()

    @staticmethod
    def _create_pixel_grid(width: int, height: int, stride: int = 1) -> np.ndarray:
        """
        Creates a dense or strided pixel coordinate grid.

        :param width: Image width.
        :param height: Image height.
        :param stride: Sampling stride. Defaults to 1 (dense grid).

        :returns: (H/stride * W/stride, 2) array of (u, v) pixel coordinates.
        """
        u_range = np.arange(0, width, stride, dtype=np.float32)
        v_range = np.arange(0, height, stride, dtype=np.float32)
        grid = np.empty((len(v_range) * len(u_range), 2), dtype=np.float32)
        grid[:, 0] = np.tile(u_range, len(v_range))
        grid[:, 1] = np.repeat(v_range, len(u_range))
        return grid
