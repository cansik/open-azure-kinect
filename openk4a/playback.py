import json
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, List, Sequence

import ffmpegio

from openk4a.capture import OpenK4ACapture
from openk4a.stream import OpenK4AVideoStream


class OpenK4APlayback:

    def __init__(self, path: Union[str, Path], loglevel: str = "quiet"):
        self._path = Path(path)

        self.loglevel = loglevel
        self._calibration_info: Optional[Dict] = None

        self.streams: List[OpenK4AVideoStream] = []

    def open(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Could not find {self._path}")

        # probe file to find out which streams are available
        stream_infos = ffmpegio.probe.streams_basic(str(self._path))

        # create stream descriptions
        self.streams.clear()
        for stream_info in stream_infos:
            if stream_info["codec_type"] == "video":
                stream = OpenK4AVideoStream(
                    index=int(stream_info["index"]),
                    codec_name=stream_info["codec_name"],
                    width=int(stream_info["width"]),
                    height=int(stream_info["height"]),
                    frame_rate=float(stream_info["r_frame_rate"]),
                    title=stream_info["tags"]["title"]
                )
                self.streams.append(stream)

            if stream_info["codec_type"] == "attachment":
                # read calibration information
                if "K4A_CALIBRATION_FILE" in stream_info["tags"]:
                    filename = stream_info["tags"]["filename"]
                    self._extract_calibration_data(filename)

    def read(self) -> OpenK4ACapture:
        pass

    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _extract_calibration_data(self, filename: str):
        output_file = Path(tempfile.gettempdir(), filename)

        # warning: this extracts only the first file, maybe the id has to be increased
        args = ["ffmpeg", "-dump_attachment:t:0", str(output_file),
                "-i", str(self._path),
                "-y", *self._loglevel_param]
        subprocess.run(args)

        if output_file.exists():
            self._calibration_info = json.loads(output_file.read_text("UTF-8"))
            output_file.unlink()
        else:
            raise FileNotFoundError("Calibration data could not been extracted.")

    @property
    def _loglevel_param(self) -> Sequence[str]:
        return "-loglevel", self.loglevel

    @property
    def path(self) -> Path:
        return self._path

    @property
    def calibration_info(self) -> Optional[Dict]:
        return self._calibration_info
