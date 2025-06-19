import argparse

import cv2

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform
from playground.utils import normalize_image, concat_images_horizontally


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.is_looping = False
    azure.open()

    transform = CameraTransform(azure.color_calibration, azure.depth_calibration)

    while capture := azure.read():
        color = capture.color
        infrared = normalize_image(capture.ir)

        # Align IR to color space
        infrared_to_color = transform.align_image_depth_to_color(infrared, capture.depth)
        ir_blended_on_color = cv2.addWeighted(color, 0.5, infrared_to_color, 0.5, 0)

        # Align color to IR space
        color_to_ir = transform.align_image_color_to_depth(color, capture.depth)
        ir_3ch = cv2.cvtColor(infrared, cv2.COLOR_GRAY2BGR)
        color_blended_on_ir = cv2.addWeighted(ir_3ch, 0.5, color_to_ir, 0.5, 0)

        # Combine horizontally for side-by-side comparison
        result = concat_images_horizontally(ir_blended_on_color, color_blended_on_ir, target_height=640)

        cv2.imshow("IR -> Color | Color -> IR", result)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    azure.close()


if __name__ == "__main__":
    main()
