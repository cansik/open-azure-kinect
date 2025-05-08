import argparse

import cv2
import numpy as np

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform, compute_distortion_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = True
    azure.open()

    depth_calibration = azure.depth_calibration
    camera_transform = CameraTransform(azure.color_calibration, depth_calibration, 1500)
    distortion_mapping = compute_distortion_mapping(depth_calibration)

    width = 640
    height = 576

    # Create a grid of UV coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv_coordinates = np.column_stack((u.ravel(), v.ravel()))

    # Apply distortion transformation to UV coordinates
    uv = distortion_mapping.transform(uv_coordinates)

    while capture := azure.read():
        depth_map = capture.depth

        # Round the UV coordinates and extract corresponding depth values
        uv_int = np.round(uv).astype(np.int32)
        depth_values = depth_map[uv_int[:, 1], uv_int[:, 0]].reshape(-1, 1)

        # Combine UV coordinates and depth values into object points
        object_points = np.hstack((uv, depth_values)).astype(np.float32)

        # Convert object points to real 3D coordinates (x, y, z)
        # Camera matrix from the depth calibration (3x3 intrinsic matrix)
        K = depth_calibration.intrinsics.camera_matrix

        # Compute 3D world coordinates using the camera matrix
        # x = (u - cx) * depth / fx
        # y = (v - cy) * depth / fy
        # z = depth
        # Where (cx, cy) is the optical center, and (fx, fy) are the focal lengths.

        # Extract intrinsic parameters from the camera matrix K
        fx = K[0, 0]  # Focal length in x direction
        fy = K[1, 1]  # Focal length in y direction
        cx = K[0, 2]  # Optical center x
        cy = K[1, 2]  # Optical center y

        # Compute real-world 3D coordinates
        x = (object_points[:, 0] - cx) * object_points[:, 2] / fx
        y = (object_points[:, 1] - cy) * object_points[:, 2] / fy
        z = object_points[:, 2]

        # Stack to form the full 3D coordinates (x, y, z)
        points_3d = np.vstack((x, y, z)).T

        # Display depth map
        cv2.imshow(f"Depth", depth_map)
        cv2.waitKey(1)  # Set to 1 for faster updates, otherwise it will wait indefinitely

    azure.close()


if __name__ == "__main__":
    main()
