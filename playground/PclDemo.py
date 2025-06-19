import argparse
import threading
from queue import Queue

import numpy as np
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform, compute_distortion_mapping

# Global Queue for thread communication
pointcloud_queue = Queue()


# Video processing thread function
def process_video(input_file, frame_rate):
    azure = OpenK4APlayback(input_file)
    azure.is_looping = True
    azure.open()

    depth_calibration = azure.depth_calibration
    camera_transform = CameraTransform(azure.color_calibration, depth_calibration)
    distortion_mapping = compute_distortion_mapping(depth_calibration)

    width = 640
    height = 576

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv_coordinates = np.column_stack((u.ravel(), v.ravel()))
    uv = distortion_mapping.transform(uv_coordinates)

    while capture := azure.read():
        depth_map = capture.depth

        # Round the UV coordinates and extract corresponding depth values
        uv_int = np.round(uv).astype(np.int32)
        depth_values = depth_map[uv_int[:, 1], uv_int[:, 0]].reshape(-1, 1)

        # Combine UV coordinates and depth values into object points
        object_points = np.hstack((uv, depth_values)).astype(np.float32)

        # Convert object points to real 3D coordinates (x, y, z)
        K = depth_calibration.intrinsics.camera_matrix

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

        # Put the points in the queue for the main thread
        pointcloud_queue.put(points_3d)

        # Frame rate control (optional)
        # time.sleep(1.0 / frame_rate)

    azure.close()


is_first_time = True


# Main function where the graphics are updated
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process a depth video file and visualize the point cloud.")
    parser.add_argument("input", type=str, help="Input MKV file.")
    parser.add_argument("--frame-rate", type=float, default=30.0, help="Frame rate to process the video.")
    args = parser.parse_args()

    # Set up the renderer and scene
    renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
    scene = gfx.Scene()

    # Create a point cloud geometry
    positions = np.zeros((640 * 576, 3)).astype(np.float32)
    geometry = gfx.Geometry(positions=positions)

    material = gfx.PointsMaterial(size=1)
    points = gfx.Points(geometry, material)
    points.local.scale = (1, -1, -1)
    scene.add(points)

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.show_object(points, scale=1)
    controller = gfx.OrbitController(camera, register_events=renderer)

    # Animation loop to update the scene and point cloud
    def animate():
        if not pointcloud_queue.empty():
            points_3d = pointcloud_queue.get()
            # Update the point cloud geometry with the new 3D points
            positions: gfx.Buffer = geometry.positions
            positions.set_data(points_3d)

            global is_first_time
            if is_first_time:
                camera.show_object(points, scale=1)
                is_first_time = False

        renderer.render(scene, camera)
        renderer.request_draw()

    # Start the video processing in a separate thread
    video_thread = threading.Thread(target=process_video, args=(args.input, args.frame_rate))
    video_thread.start()

    # Start the rendering loop
    renderer.request_draw(animate)
    run()


if __name__ == "__main__":
    main()
