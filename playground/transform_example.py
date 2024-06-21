import argparse

import cv2

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform
from playground.CharucoDetectionHelper import CharucoDetectionHelper
from playground.translation_example import normalize_image
from playground.utils import concat_images_horizontally, annotate_points


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

        annotate_points(color, color_points)
        annotate_points(infrared, depth_points)

        cv2.imshow("Result", concat_images_horizontally(color, infrared))
        cv2.waitKey(0)

    azure.close()


if __name__ == "__main__":
    main()
