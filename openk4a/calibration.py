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
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @fx.setter
    def fx(self, value: float):
        self.camera_matrix[0, 0] = value

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @fy.setter
    def fy(self, value: float):
        self.camera_matrix[1, 1] = value

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @cx.setter
    def cx(self, value: float):
        self.camera_matrix[0, 2] = value

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @cy.setter
    def cy(self, value: float):
        self.camera_matrix[1, 2] = value

    # distortion parameters
    @property
    def k1(self) -> float:
        return float(self.distortion_coefficients[0])

    @k1.setter
    def k1(self, value: float):
        self.distortion_coefficients[0] = value

    @property
    def k2(self) -> float:
        return float(self.distortion_coefficients[1])

    @k2.setter
    def k2(self, value: float):
        self.distortion_coefficients[1] = value

    @property
    def p1(self) -> float:
        return float(self.distortion_coefficients[2])

    @p1.setter
    def p1(self, value: float):
        self.distortion_coefficients[2] = value

    @property
    def p2(self) -> float:
        return float(self.distortion_coefficients[3])

    @p2.setter
    def p2(self, value: float):
        self.distortion_coefficients[3] = value

    @property
    def k3(self) -> float:
        return float(self.distortion_coefficients[4])

    @k3.setter
    def k3(self, value: float):
        self.distortion_coefficients[4] = value

    @property
    def k4(self) -> float:
        return float(self.distortion_coefficients[5])

    @k4.setter
    def k4(self, value: float):
        self.distortion_coefficients[5] = value

    @property
    def k5(self) -> float:
        return float(self.distortion_coefficients[6])

    @k5.setter
    def k5(self, value: float):
        self.distortion_coefficients[6] = value

    @property
    def k6(self) -> float:
        return float(self.distortion_coefficients[7])

    @k6.setter
    def k6(self, value: float):
        self.distortion_coefficients[7] = value


@dataclass
class Extrinsics:
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # in m, vector 1x3


@dataclass
class CameraCalibration:
    intrinsics: Intrinsics
    extrinsics: Extrinsics

    width: int
    height: int

    metric_radius: float
