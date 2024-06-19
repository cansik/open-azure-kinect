from dataclasses import dataclass

from openk4a.calibration import CameraCalibration
from openk4a.types import ColorResolution, DepthMode


@dataclass
class Size:
    width: int
    height: int


@dataclass
class CameraCalibrationModeInfo:
    calibration_image_binned_resolution: Size
    crop_offset: Size
    output_image_resolution: Size


def transformation_get_mode_specific_camera_calibration(calibration: CameraCalibration,
                                                        mode_info: CameraCalibrationModeInfo,
                                                        pixelized_zero_centered_output: bool) -> CameraCalibration:
    if not (mode_info.calibration_image_binned_resolution.width > 0 and
            mode_info.calibration_image_binned_resolution.height > 0 and
            mode_info.output_image_resolution.width > 0 and
            mode_info.output_image_resolution.height > 0):
        raise ValueError("Expect calibration image binned resolution and output image resolution are larger than 0")

    mode_specific_camera_calibration = CameraCalibration(
        calibration.intrinsics,
        calibration.extrinsics,
        calibration.width,
        calibration.height,
        calibration.metric_radius
    )

    params = mode_specific_camera_calibration.intrinsics

    cx = params.cx * mode_info.calibration_image_binned_resolution.width
    cy = params.cy * mode_info.calibration_image_binned_resolution.height
    fx = params.fx * mode_info.calibration_image_binned_resolution.width
    fy = params.fy * mode_info.calibration_image_binned_resolution.height

    cx -= mode_info.crop_offset.width
    cy -= mode_info.crop_offset.height

    if pixelized_zero_centered_output:
        params.cx = cx - 0.5
        params.cy = cy - 0.5
        params.fx = fx
        params.fy = fy
    else:
        params.cx = cx / mode_info.output_image_resolution.width
        params.cy = cy / mode_info.output_image_resolution.height
        params.fx = fx / mode_info.output_image_resolution.width
        params.fy = fy / mode_info.output_image_resolution.height

    mode_specific_camera_calibration.width = mode_info.output_image_resolution.width
    mode_specific_camera_calibration.height = mode_info.output_image_resolution.height

    return mode_specific_camera_calibration


def transformation_get_mode_specific_depth_camera_calibration(calibration: CameraCalibration,
                                                              depth_mode: DepthMode) -> CameraCalibration:
    if calibration is None:
        raise ValueError("Calibration data is required")

    if not (calibration.width == 1024 and calibration.height == 1024):
        raise ValueError("Unexpected raw camera calibration resolution, should be (1024, 1024)")

    if depth_mode == DepthMode.NFOV_2X2BINNED:
        mode_info = CameraCalibrationModeInfo(Size(512, 512), Size(96, 90), Size(320, 288))
    elif depth_mode == DepthMode.NFOV_UNBINNED:
        mode_info = CameraCalibrationModeInfo(Size(1024, 1024), Size(192, 180), Size(640, 576))
    elif depth_mode == DepthMode.WFOV_2X2BINNED:
        mode_info = CameraCalibrationModeInfo(Size(512, 512), Size(0, 0), Size(512, 512))
    elif depth_mode in [DepthMode.WFOV_UNBINNED, DepthMode.PASSIVE_IR]:
        mode_info = CameraCalibrationModeInfo(Size(1024, 1024), Size(0, 0), Size(1024, 1024))
    else:
        raise ValueError("Unexpected depth mode")

    return transformation_get_mode_specific_camera_calibration(calibration, mode_info, True)


def transformation_get_mode_specific_color_camera_calibration(calibration: CameraCalibration,
                                                              color_resolution: ColorResolution) -> CameraCalibration:
    if calibration.width * 9 / 16 == calibration.height:
        mode_info = CameraCalibrationModeInfo(Size(4096, 2304), Size(0, -384), Size(4096, 3072))
        return transformation_get_mode_specific_camera_calibration(
            calibration, mode_info, False)
    elif calibration.width * 3 / 4 == calibration.height:
        mode_specific_camera_calibration = CameraCalibration(
            calibration.intrinsics,
            calibration.extrinsics,
            calibration.width,
            calibration.height,
            calibration.metric_radius
        )
    else:
        raise ValueError("Unexpected aspect ratio, should either be 16:9 or 4:3")

    if color_resolution == ColorResolution.RESOLUTION_720P:
        mode_info = CameraCalibrationModeInfo(Size(1280, 960), Size(0, 120), Size(1280, 720))
    elif color_resolution == ColorResolution.RESOLUTION_1080P:
        mode_info = CameraCalibrationModeInfo(Size(1920, 1440), Size(0, 180), Size(1920, 1080))
    elif color_resolution == ColorResolution.RESOLUTION_1440P:
        mode_info = CameraCalibrationModeInfo(Size(2560, 1920), Size(0, 240), Size(2560, 1440))
    elif color_resolution == ColorResolution.RESOLUTION_1536P:
        mode_info = CameraCalibrationModeInfo(Size(2048, 1536), Size(0, 0), Size(2048, 1536))
    elif color_resolution == ColorResolution.RESOLUTION_2160P:
        mode_info = CameraCalibrationModeInfo(Size(3840, 2880), Size(0, 360), Size(3840, 2160))
    elif color_resolution == ColorResolution.RESOLUTION_3072P:
        mode_info = CameraCalibrationModeInfo(Size(4096, 3072), Size(0, 0), Size(4096, 3072))
    else:
        raise ValueError(f"Unexpected color resolution type {color_resolution}")

    return transformation_get_mode_specific_camera_calibration(
        mode_specific_camera_calibration, mode_info, True
    )
