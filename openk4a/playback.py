import json
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, List, Sequence, Iterator

import ffmpegio
import numpy as np
from ffmpegio.streams import AviMediaReader

from openk4a.capture import OpenK4ACapture
from openk4a.stream import OpenK4AVideoStream, OpenK4AColorStreamName, OpenK4ADepthStreamName, OpenK4AInfraredStreamName


class OpenK4APlayback:

    def __init__(self, path: Union[str, Path], is_looping: bool = False, loglevel: str = "quiet"):
        self._path = Path(path)
        self.is_looping = is_looping

        self.loglevel = loglevel
        self._calibration_info: Optional[Dict] = None

        self.streams: List[OpenK4AVideoStream] = []
        self._stream_map: Dict[str, OpenK4AVideoStream] = {}

        self.block_size: int = 1
        self._video_reader: Optional[AviMediaReader] = None
        self._frame_iterator: Optional[Iterator] = None

    def open(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Could not find {self._path}")

        self._extract_stream_infos()

        # create video reader
        streams = [s.stream_name for s in self.streams]
        pix_fmts = {f"pix_fmt:{s.stream_name}": s.pixel_format for s in self.streams}

        self._video_reader = AviMediaReader(str(self._path),
                                            map=streams,
                                            blocksize=self.block_size,
                                            **pix_fmts)
        self._frame_iterator = iter(self._video_reader)

    def read(self) -> Optional[OpenK4ACapture]:
        frames: Optional[Dict[str, np.ndarray]] = next(self._frame_iterator, None)

        # handle if no frame could be read
        if frames is None:
            if not self.is_looping:
                return None

            # create new iterator to loop through frames
            self._frame_iterator = iter(self._video_reader)
            frames = next(self._frame_iterator, None)

        capture = OpenK4ACapture()

        for stream_name, data in frames.items():
            stream = self._stream_map[stream_name]

            if stream.title == OpenK4AColorStreamName:
                capture.color = data[0]
            elif stream.title == OpenK4ADepthStreamName:
                capture.depth = data[0]
            elif stream.title == OpenK4AInfraredStreamName:
                capture.ir = data[0]

        return capture

    def close(self):
        self._video_reader.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _extract_stream_infos(self):
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

        self._stream_map.clear()
        self._stream_map = {s.stream_name: s for s in self.streams}

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
