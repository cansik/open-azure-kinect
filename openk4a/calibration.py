from dataclasses import dataclass
from enum import Enum

import numpy as np


class CameraType(Enum):
    Depth = "CALIBRATION_CameraPurposeDepth"
    Color = "CALIBRATION_CameraPurposePhotoVideo"


@dataclass
class Intrinsics:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    @property
    def fx(self):
        return self.camera_matrix[0, 0]

    @fx.setter
    def fx(self, value):
        self.camera_matrix[0, 0] = value

    @property
    def fy(self):
        return self.camera_matrix[1, 1]

    @fy.setter
    def fy(self, value):
        self.camera_matrix[1, 1] = value

    @property
    def cx(self):
        return self.camera_matrix[0, 2]

    @cx.setter
    def cx(self, value):
        self.camera_matrix[0, 2] = value

    @property
    def cy(self):
        return self.camera_matrix[1, 2]

    @cy.setter
    def cy(self, value):
        self.camera_matrix[1, 2] = value


@dataclass
class Extrinsics:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class CameraCalibration:
    intrinsics: Intrinsics
    extrinsics: Extrinsics

    sensor_width: int
    sensor_height: int

    metric_radius: float
