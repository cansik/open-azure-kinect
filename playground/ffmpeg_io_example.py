import argparse
from pathlib import Path

import cv2
import numpy as np
import ffmpegio
import ffmpegio.filtergraph as fgb


def read_rgb(input_path: Path):
    with ffmpegio.open(str(input_path), 'rv', blocksize=1, pix_fmt='rgb24') as fin:
        for frames in fin:
            for frame in frames:
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)


def read_infrared(input_path: Path):
    with ffmpegio.open(str(input_path), 'rv', blocksize=1, map="0:2", pix_fmt='gray16le') as fin:
        for frames in fin:
            for frame in frames:
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                max_value = 500
                normalize = (np.clip(frame.astype(float), 0, max_value) / max_value * 255).astype(np.uint8)

                cv2.imshow("Frame", normalize)
                cv2.waitKey(1)


def read_all(input_path: Path):
    streams = ["v:0", "v:1", "v:2"]
    pix_fmts = {"pix_fmt:v:0": "rgb24", "pix_fmt:v:1": "gray16le", "pix_fmt:v:2": "gray16le"}

    with ffmpegio.open(str(input_path), "rvv", map=streams, blocksize=1, pix_fmt="rgb24", **pix_fmts) as fin:
        for frames in fin:
            print(list(frames.keys()))

            v0: np.ndarray = frames["v:0"]

            v2: np.ndarray = frames["v:2"]
            data = v2.reshape(-1).view(np.uint16).reshape([-1, 576, 960, 1])

            v2_bytes = v2.tobytes()
            v2_mat = np.frombuffer(v2_bytes, np.uint16).reshape([-1, 576, 640, 1])

            max_value = 500
            normalize = (np.clip(v2_mat.astype(float), 0, max_value) / max_value * 255).astype(np.uint8)

            cv2.imshow("Frame", normalize[0])
            cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    input_path = Path(args.input)

    info = ffmpegio.probe.streams_basic(str(input_path))

    # read_rgb(input_path)
    # read_infrared(input_path)
    read_all(input_path)


if __name__ == "__main__":
    main()
