import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

import ffmpeg
import numpy as np


def read_frames(file_path: str,
                width: int, height: int, channels: int,
                pixel_format: str, data_format: int,
                mapping: str = "0:0",
                stream_time: str = "00:00:00", frame_count: int = 1) -> np.ndarray:
    out, err = (
        ffmpeg
        .input(file_path, ss=stream_time)
        # .filter_("select", f"eq(n,{10})")
        .output("pipe:", format="rawvideo", pix_fmt=pixel_format, vsync=0, vframes=frame_count, map=mapping)
        .run(capture_stdout=True)
    )
    return np.frombuffer(out, data_format).reshape([-1, height, width, channels])


def read_calibration(file_path: str, stream_info: Dict[str, Any]) -> Dict[str, Any]:
    args = ["ffmpeg", "-dump_attachment:t", "", "-i", file_path, "-y"]
    subprocess.run(args)
    output_file = Path(stream_info["tags"]["filename"])

    if output_file.exists():
        data = json.loads(output_file.read_text("UTF-8"))
        output_file.unlink()
        return data
    else:
        raise FileNotFoundError("Calibration data could not been extracted.")


def display_frames(input_path: Path):
    probe = ffmpeg.probe(str(input_path))
    streams = probe["streams"]

    for stream in streams:
        print(stream)

    # read rgb
    rgb_stream = streams[0]
    rgb_width = rgb_stream["width"]
    rgb_height = rgb_stream["height"]
    rgb_frames = read_frames(str(input_path), rgb_width, rgb_height, 3, "rgb24", np.uint8)

    # read depth
    depth_stream = streams[1]
    depth_width = depth_stream["width"]
    depth_height = depth_stream["height"]
    depth_frames = read_frames(str(input_path), depth_width, depth_height, 1, "gray16le", np.uint16, mapping="0:1")

    # read ir (same width)
    ir_stream = streams[2]
    ir_frames = read_frames(str(input_path), depth_width, depth_height, 1, "gray16le", np.uint16, mapping="0:2")

    # read calibration data
    calibration = read_calibration(str(input_path), streams[3])

    # read imu data

    print(rgb_stream)


def read_combined(input_path: Path):
    depth_pipe = "openk4a-depth"
    ir_pipe = "openk4a-ir"

    # os.mkfifo(depth_pipe)
    # os.mkfifo(ir_pipe)

    args = [
        "ffmpeg -i input.mkv",
        ""
    ]
    command = " ".join(args).strip()
    print(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    read_combined(input_path)
    # display_frames(input_path)


if __name__ == "__main__":
    main()
