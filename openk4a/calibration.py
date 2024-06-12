from dataclasses import dataclass
from enum import Enum

import numpy as np


class CameraType(Enum):
    Depth = "CALIBRATION_CameraPurposeDepth",
    Color = "CALIBRATION_CameraPurposePhotoVideo"


@dataclass
class Intrinsics:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray


@dataclass
class Extrinsics:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class CameraCalibration:
    intrinsics: Intrinsics
    extrinsics: Extrinsics

    metric_radius: float
