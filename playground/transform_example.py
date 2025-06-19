import argparse

import cv2

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform
from playground.CharucoDetectionHelper import CharucoDetectionHelper
from playground.utils import annotate_points, normalize_image, concat_images_horizontally, compute_projection_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = False
    azure.open()

    detector = CharucoDetectionHelper()

    transform = CameraTransform(azure.color_calibration, azure.depth_calibration)

    while capture := azure.read():
        color = capture.color
        infrared = cv2.cvtColor(normalize_image(capture.ir), cv2.COLOR_GRAY2BGR)

        color_detections = detector.detect_markers(color)
        color_points = color_detections.corners[:, :, 0].reshape(-1, 2)

        ir_detections = detector.detect_markers(infrared)
        ir_points = ir_detections.corners[:, :, 0].reshape(-1, 2)

        color_points_est = transform.transform_2d_depth_to_color(ir_points, capture.depth)
        ir_points_est = transform.transform_2d_color_to_depth(color_points, capture.depth)

        d2c_count, d2c_error = compute_projection_error(ir_detections, color_detections, color_points_est)
        c2d_count, c2d_error = compute_projection_error(color_detections, ir_detections, ir_points_est)

        print(f"Depth to Color Error: {d2c_error:.4f} (using {d2c_count} points)")
        print(f"Color to Depth Error: {c2d_error:.4f} (using {c2d_count} points)")

        ir_d2c = infrared.copy()
        color_d2c = color.copy()

        ir_c2d = infrared.copy()
        color_c2d = color.copy()

        annotate_points(ir_d2c, ir_points)
        annotate_points(color_d2c, color_points_est)

        annotate_points(ir_c2d, ir_points_est)
        annotate_points(color_c2d, color_points)

        cv2.imshow("Result D2C", concat_images_horizontally(color_d2c, ir_d2c, target_height=640))
        cv2.imshow("Result C2D", concat_images_horizontally(color_c2d, ir_c2d, target_height=640))
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    azure.close()


if __name__ == "__main__":
    main()
