import argparse

import cv2

from openk4a.playback import OpenK4APlayback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.open()

    for stream in azure.streams:
        print(stream)

    print(azure.calibration_info)

    while capture := azure.read():
        cv2.imshow("Demo", cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    azure.close()


if __name__ == "__main__":
    main()
