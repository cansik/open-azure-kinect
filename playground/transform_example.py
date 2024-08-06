import argparse

import cv2
import numpy as np

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform
from playground.CharucoDetectionHelper import CharucoDetectionHelper
from playground.utils import concat_images_horizontally, annotate_points, normalize_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = True
    azure.open()

    detector = CharucoDetectionHelper()

    while capture := azure.read():
        center_depth = int(capture.depth[576 // 2, 640 // 2])

        transform = CameraTransform(azure.color_calibration, azure.depth_calibration, center_depth)

        color = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
        infrared = cv2.cvtColor(normalize_image(capture.ir), cv2.COLOR_GRAY2BGR)

        detections = detector.detect_markers(color)
        color_points = detections.corners[:, :, 0].reshape(-1, 2)

        depth_points = transform.transform_2d_color_to_depth(color_points)

        # opencv convertions
        depth_values = np.full((len(color_points), 1), center_depth)
        depth_points_sp = transform.transform_2d_color_to_depth_cv2(color_points, depth_values)
        depth_points_optimise_sp = transform.optimized_transform_2d_color_to_depth_cv2(color_points, depth_values, capture.depth)
        color_points_sp = transform.transform_2d_depth_to_color_cv2(depth_points_sp, capture.depth)

        infrared2 = infrared.copy()
        infrared2_opt = infrared.copy()
        infrared3 = infrared.copy()

        color_inv = color.copy()
        infrared_inv = infrared3.copy()

        detections = detector.detect_markers(infrared_inv)
        depth_points_inv = detections.corners[:, :, 0].reshape(-1, 2)

        color_points_inv = transform.transform_2d_depth_to_color_cv2(depth_points_inv, capture.depth)

        annotate_points(color, color_points)
        annotate_points(infrared, depth_points)
        annotate_points(infrared2, depth_points_sp)
        annotate_points(infrared2_opt, depth_points_optimise_sp)

        annotate_points(infrared_inv, depth_points_inv)
        annotate_points(color_inv, color_points_inv)

        cv2.imshow("Result", concat_images_horizontally(color, infrared, target_height=640))
        cv2.imshow("Reversed", concat_images_horizontally(infrared_inv, color_inv, target_height=640))
        cv2.imshow("Result SP", concat_images_horizontally(infrared2, infrared2_opt, target_height=640))
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        for i in range(300 - 10, 800 + 10, 10):
            image = infrared3.copy()
            depth_values = np.full((len(color_points), 1), i)
            depth_points_sp = transform.transform_2d_color_to_depth_cv2(color_points, depth_values)

            annotate_points(image, depth_points_sp)

            cv2.destroyAllWindows()
            cv2.imshow(f"Depth={i}", image)
            cv2.waitKey(0)

    azure.close()


if __name__ == "__main__":
    main()
