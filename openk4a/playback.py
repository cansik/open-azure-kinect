import json
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, List, Sequence, Iterator, Any

import ffmpegio
import numpy as np
from ffmpegio.streams import AviMediaReader

from openk4a.calibration import CameraType, CameraCalibration, Intrinsics, Extrinsics
from openk4a.capture import OpenK4ACapture
from openk4a.stream import OpenK4AVideoStream, OpenK4AColorStreamName, OpenK4ADepthStreamName, OpenK4AInfraredStreamName
from openk4a.types import ColorResolution, DepthMode
from openk4a.utils._calibration_utils import transformation_get_mode_specific_color_camera_calibration, \
    transformation_get_mode_specific_depth_camera_calibration


class OpenK4APlayback:

    def __init__(self, path: Union[str, Path], is_looping: bool = False, loglevel: str = "quiet"):
        self._path = Path(path)
        self.is_looping = is_looping

        self.loglevel = loglevel
        self._calibration_raw: Optional[Dict] = None

        self.streams: List[OpenK4AVideoStream] = []
        self._stream_map: Dict[str, OpenK4AVideoStream] = {}

        self.block_size: Optional[int] = 1
        self._video_reader: Optional[AviMediaReader] = None
        self._frame_iterator: Optional[Iterator] = None

        self._calibrations: Dict[CameraType, CameraCalibration] = {}

        self._depth_mode: DepthMode = DepthMode.OFF
        self._color_resolution: ColorResolution = ColorResolution.OFF

    def open(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Could not find {self._path}")

        self._extract_stream_infos()

        # create video reader
        streams = [s.stream_name for s in self.streams]
        pix_fmts = {f"pix_fmt:{s.stream_name}": s.pixel_format for s in self.streams}

        options: Dict[str, Any] = {
            **pix_fmts
        }

        if self.is_looping:
            options["stream_loop_in"] = -1

        self._video_reader = AviMediaReader(str(self._path),
                                            map=streams,
                                            blocksize=self.block_size,
                                            **options)
        self._frame_iterator = iter(self._video_reader)

    def read(self) -> Optional[OpenK4ACapture]:
        frames: Optional[Dict[str, np.ndarray]] = next(self._frame_iterator, None)

        # handle if no frame could be read
        if frames is None:
            if not self.is_looping:
                return None

            frames = next(self._frame_iterator, None)

            if frames is None:
                return None

        capture = OpenK4ACapture()

        for stream_name, data in frames.items():
            stream = self._stream_map[stream_name]

            if stream.title == OpenK4AColorStreamName:
                capture.color = data[0]
            elif stream.title == OpenK4ADepthStreamName:
                capture.depth = data[0].squeeze()
            elif stream.title == OpenK4AInfraredStreamName:
                capture.ir = data[0].squeeze()

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

                # read resolution / depth mode
                tags = stream_info["tags"]
                if "K4A_COLOR_MODE" in tags:
                    self._color_resolution = ColorResolution(stream.height)
                if "K4A_DEPTH_MODE" in tags:
                    self._depth_mode = DepthMode[tags["K4A_DEPTH_MODE"]]

            if stream_info["codec_type"] == "attachment":
                # read calibration information
                if "K4A_CALIBRATION_FILE" in stream_info["tags"]:
                    filename = stream_info["tags"]["filename"]
                    self._extract_calibration_data(filename)

        self._stream_map.clear()
        self._stream_map = {s.stream_name: s for s in self.streams}

        self._extract_calibration_info()

    def _extract_calibration_data(self, filename: str):
        output_file = Path(tempfile.gettempdir(), filename)

        # warning: this extracts only the first file, maybe the id has to be increased
        args = ["ffmpeg", "-dump_attachment:t:0", str(output_file),
                "-i", str(self._path),
                "-y", *self._loglevel_param]
        subprocess.run(args)

        if output_file.exists():
            self._calibration_raw = json.loads(output_file.read_text(encoding="UTF-8"))
            output_file.unlink()
        else:
            raise FileNotFoundError("Calibration data could not been extracted.")

    def _extract_calibration_info(self):
        camera_types = {member.value: member for member in CameraType}

        for cam_info in self._calibration_raw["CalibrationInformation"]["Cameras"]:
            purpose = cam_info["Purpose"]
            camera_type = camera_types[purpose] if purpose in camera_types else None

            if camera_type is None:
                continue

            params = cam_info["Intrinsics"]["ModelParameters"]

            # sensor size
            sensor_width = int(cam_info["SensorWidth"])
            sensor_height = int(cam_info["SensorHeight"])

            # extract intrinsic parameters
            cx, cy, fx, fy = np.array(params[:4])

            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # distortion coefficients in opencv-compatible format
            # https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/structk4a__calibration__intrinsic__parameters__t_1_1__param.html
            # k1, k2, p1, p2, k3, k4, k5, k6
            distortion_coefficients = np.array([params[4], params[5], params[13], params[12], *params[6:10]],
                                               dtype=np.float32)
            metric_radius = float(cam_info["MetricRadius"])

            # metric_radius equals 0 means that calibration failed to estimate this parameter
            if metric_radius < 0.0001:
                metric_radius = 1.7

            # extract extrinsic parameters
            rotation = np.array(cam_info["Rt"]["Rotation"], dtype=np.float32).reshape(3, 3)
            translation = np.array(cam_info["Rt"]["Translation"], dtype=np.float32).reshape(1, 3)

            # millimeter to meter conversion
            translation = translation / 1000

            raw_calibration = CameraCalibration(
                Intrinsics(camera_matrix, distortion_coefficients),
                Extrinsics(rotation, translation),
                sensor_width, sensor_height,
                metric_radius
            )

            if camera_type == CameraType.Color:
                calib = transformation_get_mode_specific_color_camera_calibration(raw_calibration,
                                                                                  self.color_resolution)
            elif camera_type == CameraType.Depth:
                calib = transformation_get_mode_specific_depth_camera_calibration(raw_calibration,
                                                                                  self.depth_mode)
            else:
                raise ValueError("Unsupported camera type.")

            self._calibrations[camera_type] = calib

    @property
    def _loglevel_param(self) -> Sequence[str]:
        return "-loglevel", self.loglevel

    @property
    def path(self) -> Path:
        return self._path

    @property
    def calibration_raw(self) -> Optional[Dict]:
        return self._calibration_raw

    def get_calibration(self, camera_type: CameraType) -> Optional[CameraCalibration]:
        if camera_type in self._calibrations:
            return self._calibrations[camera_type]
        return None

    @property
    def color_calibration(self) -> Optional[CameraCalibration]:
        return self.get_calibration(camera_type=CameraType.Color)

    @property
    def depth_calibration(self) -> Optional[CameraCalibration]:
        return self.get_calibration(camera_type=CameraType.Depth)

    @property
    def depth_mode(self) -> DepthMode:
        return self._depth_mode

    @property
    def color_resolution(self) -> ColorResolution:
        return self._color_resolution
