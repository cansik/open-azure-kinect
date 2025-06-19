import numpy as np

from openk4a.calibration import Intrinsics, Extrinsics, CameraCalibration

# Color calibration data
color_intrinsics: Intrinsics = Intrinsics(
    camera_matrix=np.array([
        [917.8631, 0.0, 953.6561],
        [0.0, 917.3132, 553.73157],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32),
    distortion_coefficients=np.array([
        5.2921218e-01, -2.5190108e+00, 4.4136654e-04,
        4.9181699e-07, 1.3768097e+00, 4.0784040e-01,
        -2.3492301e+00, 1.3108439e+00
    ], dtype=np.float32)
)

color_extrinsics: Extrinsics = Extrinsics(
    rotation=np.array([
        [9.99999166e-01, 3.00158339e-04, 1.23397936e-03],
        [-4.22734156e-04, 9.94930208e-01, 1.00566655e-01],
        [-1.19753752e-03, -1.00567095e-01, 9.94929552e-01]
    ], dtype=np.float32),
    translation=np.array([[-0.03212256, -0.0019677, 0.00406363]], dtype=np.float32)
)

color_camera_calibration: CameraCalibration = CameraCalibration(
    intrinsics=color_intrinsics,
    extrinsics=color_extrinsics,
    width=1920,
    height=1080,
    metric_radius=1.7
)

# Depth calibration data
depth_intrinsics: Intrinsics = Intrinsics(
    camera_matrix=np.array([
        [504.3556, 0.0, 331.07727],
        [0.0, 504.46213, 333.14966],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32),
    distortion_coefficients=np.array([
        5.5100384e+00, 3.6645205e+00, -9.9567362e-05,
        -8.8820896e-05, 1.8916880e-01, 5.8385630e+00,
        5.4994097e+00, 9.9640810e-01
    ], dtype=np.float32)
)

depth_extrinsics: Extrinsics = Extrinsics(
    rotation=np.eye(3, dtype=np.float32),
    translation=np.zeros((1, 3), dtype=np.float32)
)

depth_camera_calibration: CameraCalibration = CameraCalibration(
    intrinsics=depth_intrinsics,
    extrinsics=depth_extrinsics,
    width=640,
    height=576,
    metric_radius=1.739999771118164
)

# List of all calibrations for testing
camera_calibrations: list[CameraCalibration] = [
    color_camera_calibration,
    depth_camera_calibration
]
