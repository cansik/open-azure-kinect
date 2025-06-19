import argparse
import threading
from queue import Queue

import numpy as np
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform

STRIDE = 1
PCL_WIDTH = 640 // STRIDE
PCL_HEIGHT = 576 // STRIDE

# Global Queue for thread communication
pointcloud_queue = Queue()


def create_uv_samples(width: int = 640, height: int = 576, stride: int = 1) -> np.ndarray:
    u_vals = np.arange(0, width, stride, dtype=np.float32)
    v_vals = np.arange(0, height, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u_vals, v_vals)  # shapes (H/stride, W/stride)
    return np.stack([uu, vv], axis=-1).reshape(-1, 2)


# Video processing thread function
def process_video(input_file, frame_rate):
    azure = OpenK4APlayback(input_file)
    azure.is_looping = True
    azure.open()

    depth_calibration = azure.depth_calibration
    camera_transform = CameraTransform(azure.color_calibration, depth_calibration)

    # pre-calculate uv samples
    samples = create_uv_samples(stride=STRIDE)

    while capture := azure.read():
        depth_map = capture.depth

        # slow way to create pointcloud, because pixel buffer is not re-used
        # point_cloud = camera_transform.create_pointcloud(depth_map, stride=STRIDE)

        # faster to re-use samples
        point_cloud = camera_transform.transform_depth_to_3d(samples, depth_map)

        # put the points in the queue for the main thread
        pointcloud_queue.put(point_cloud)

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
    positions = np.zeros((PCL_WIDTH * PCL_HEIGHT, 3)).astype(np.float32)
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
