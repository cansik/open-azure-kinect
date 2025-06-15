import argparse

import cv2

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

        transform = CameraTransform(azure.color_calibration, azure.depth_calibration)

        color = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
        infrared = cv2.cvtColor(normalize_image(capture.ir), cv2.COLOR_GRAY2BGR)

        color_detections = detector.detect_markers(color)
        color_points = color_detections.corners[:, :, 0].reshape(-1, 2)

        ir_detections = detector.detect_markers(infrared)
        ir_points = ir_detections.corners[:, :, 0].reshape(-1, 2)

        color_points_est = transform.transform_2d_depth_to_color(ir_points, capture.depth)

        ir_preview = infrared.copy()
        color_preview = color.copy()

        annotate_points(ir_preview, ir_points)
        annotate_points(color_preview, color_points_est)

        cv2.imshow("Result D2C", concat_images_horizontally(color_preview, ir_preview, target_height=640))
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    azure.close()


if __name__ == "__main__":
    main()
