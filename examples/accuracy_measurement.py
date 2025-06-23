import argparse
import ctypes
from abc import ABC, abstractmethod
from ctypes import (
    c_void_p, c_int, c_size_t,
    c_float, POINTER, byref
)
from dataclasses import dataclass
from typing import List
from typing import Optional, Tuple

import cv2
import numpy as np
from pyk4a.calibration import Calibration as PyK4ACalibration, CalibrationType
from pyk4a.capture import PyK4ACapture
from pyk4a.playback import PyK4APlayback

from openk4a.playback import OpenK4APlayback
from openk4a.transform import CameraTransform as OpenK4ATransform


@dataclass
class Config:
    input_file: str
    sample_spacing: int = 20
    margin: int = 50


class AzureKinect(ABC):
    def __init__(self, filename: str):
        self.filename = filename

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read_capture(self):
        ...

    @abstractmethod
    def get_color_size(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_depth_size(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def transform_d2c(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def transform_c2d(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class OpenK4AAdapter(AzureKinect):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.playback = OpenK4APlayback(filename)

    def open(self) -> None:
        self.playback.is_looping = False
        self.playback.open()
        self.transform = OpenK4ATransform(
            self.playback.color_calibration,
            self.playback.depth_calibration
        )

    def read_capture(self):
        return self.playback.read()

    def get_color_size(self) -> Tuple[int, int]:
        c = self.playback.color_calibration
        return c.width, c.height

    def get_depth_size(self) -> Tuple[int, int]:
        d = self.playback.depth_calibration
        return d.width, d.height

    def transform_d2c(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        out = []
        for u, v in points:
            try:
                p = self.transform.transform_2d_depth_to_color(np.array([[u, v]]), depth)[0]
            except Exception:
                p = (np.nan, np.nan)
            out.append(p)
        return np.array(out, dtype=np.float32)

    def transform_c2d(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        out = []
        for u, v in points:
            try:
                p = self.transform.transform_2d_color_to_depth(np.array([[u, v]]), depth)[0]
            except Exception:
                p = (np.nan, np.nan)
            out.append(p)
        return np.array(out, dtype=np.float32)

    def close(self) -> None:
        self.playback.close()


class PyK4AAdapter(AzureKinect):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.playback = PyK4APlayback(filename)
        self._capture: Optional[PyK4ACapture] = None

        # bmonkey‐patch color to depth here
        self._patch_color2d_to_depth2d()

    def _patch_color2d_to_depth2d(self):
        lib = ctypes.CDLL("k4a.dll")

        create_buf = lib.k4a_image_create_from_buffer
        create_buf.argtypes = [
            c_int, c_int, c_int, c_int,
            c_void_p, c_size_t,
            c_void_p, c_void_p,
            POINTER(c_void_p)
        ]
        create_buf.restype = c_int

        release_img = lib.k4a_image_release
        release_img.argtypes = [c_void_p]
        release_img.restype = None

        fn = lib.k4a_calibration_color_2d_to_depth_2d
        fn.argtypes = [
            c_void_p,
            POINTER(c_float * 2),
            c_void_p,
            POINTER(c_float * 2),
            POINTER(c_int)
        ]
        fn.restype = c_int

        K4A_IMAGE_FORMAT_DEPTH16 = 6

        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
        PyCapsule_GetPointer.restype = c_void_p

        def color2d_to_depth2d(source_pt: Tuple[float, float],
                               depth_np: np.ndarray
                               ) -> Tuple[int, Tuple[float, float], bool]:
            """
            Map a color‐pixel (u,v) into depth‐image pixel coordinates.

            Returns:
              result_code: int         # 0 == K4A_RESULT_SUCCEEDED, non‐zero == failure
              (x_depth, y_depth): Tuple[float, float]
              valid: bool              # True if projection falls inside the depth image
            """
            src = (c_float * 2)(*source_pt)
            tgt = (c_float * 2)()
            valid = c_int(0)

            h, w = depth_np.shape
            stride = w * 2
            buf_ptr = depth_np.ctypes.data_as(c_void_p)
            img_handle = c_void_p()
            rc = create_buf(
                K4A_IMAGE_FORMAT_DEPTH16,
                w, h, stride,
                buf_ptr,
                w * h * 2,
                None, None,
                byref(img_handle)
            )
            if rc != 0:
                raise RuntimeError(f"k4a_image_create_from_buffer failed ({rc})")

            try:
                capsule = self.calib._calibration_handle
                calib_ptr = PyCapsule_GetPointer(capsule, b"pyk4a calibration handle")
                res = fn(
                    calib_ptr,
                    byref(src),
                    img_handle,
                    byref(tgt),
                    byref(valid)
                )
            finally:
                release_img(img_handle)

            return res, (tgt[0], tgt[1]), bool(valid.value)

        self.color2d_to_depth2d = color2d_to_depth2d

    def open(self) -> None:
        self.playback.open()
        self.calib: PyK4ACalibration = self.playback.calibration

    def read_capture(self):
        try:
            self._capture = self.playback.get_next_capture()
            return self._capture
        except EOFError:
            return None

    def get_color_size(self) -> Tuple[int, int]:
        img = cv2.imdecode(self._capture.color, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        return w, h

    def get_depth_size(self) -> Tuple[int, int]:
        h, w = self._capture.depth.shape[:2]
        return w, h

    def transform_d2c(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        out = []
        for u, v in points:
            try:
                z = float(depth[int(v), int(u)])
                p3 = self.calib.convert_2d_to_3d((u, v), z, CalibrationType.DEPTH)
                p3c = self.calib.depth_to_color_3d(p3)
                p = self.calib.convert_3d_to_2d(p3c, CalibrationType.COLOR)
            except Exception:
                p = (np.nan, np.nan)
            out.append(p)
        return np.array(out, dtype=np.float32)

    def transform_c2d(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        out = []
        for u, v in points:
            error_code, p, success = self.color2d_to_depth2d((u, v), depth)
            if error_code != 0 or not success:
                p = (np.nan, np.nan)
            out.append(p)
        return np.array(out, dtype=np.float32)

    def close(self) -> None:
        self.playback.close()


def sample_grid(size: Tuple[int, int], spacing: int, margin: int) -> np.ndarray:
    w, h = size
    xs = np.arange(margin, w - margin, spacing)
    ys = np.arange(margin, h - margin, spacing)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)


def compute_comparison(
        ref_pts: np.ndarray,
        test_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(ref_pts).any(axis=1) & ~np.isnan(test_pts).any(axis=1)
    errors = np.linalg.norm(ref_pts[valid] - test_pts[valid], axis=1)
    return valid, errors


def draw_error_map(
        base_img: np.ndarray,
        proj_pts: np.ndarray,
        valid: np.ndarray,
        errors: np.ndarray,
        win_name: str
) -> None:
    vis = base_img.copy()
    norm = np.zeros(valid.shape[0], dtype=np.uint8)
    if errors.size > 0:
        e_min, e_max = errors.min(), errors.max()
        norm_vals = ((errors - e_min) / (e_max - e_min + 1e-6) * 255).astype(np.uint8)
        norm[valid] = norm_vals
    gray = norm.reshape(-1, 1)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET).reshape(-1, 3)
    # draw only valid
    for idx, res in enumerate(proj_pts):
        if valid[idx]:
            u, v = res.astype(int)
            color = tuple(int(c) for c in colored[idx])
            cv2.circle(vis, (u, v), 3, color=color, thickness=-1)
    # legend
    bar_h, bar_w = 20, 256
    grad = np.linspace(0, 255, bar_w, dtype=np.uint8)
    grad = np.tile(grad, (bar_h, 1))
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    h_img = vis.shape[0]
    x_off, y_off = 10, h_img - bar_h - 10
    vis[y_off:y_off + bar_h, x_off:x_off + bar_w] = grad_color
    if errors.size > 0:
        cv2.putText(vis, f"{e_min:.5f}", (x_off, y_off - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, f"{e_max:.5f}", (x_off + bar_w - 50, y_off - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    cv2.imshow(win_name, vis)


def main():
    parser = argparse.ArgumentParser(description="Compare PyK4A vs OpenK4A accuracy")
    parser.add_argument("input", help="MKV file path")
    parser.add_argument("--spacing", type=int, default=20)
    parser.add_argument("--margin", type=int, default=50)
    args = parser.parse_args()

    cfg = Config(args.input, sample_spacing=args.spacing, margin=args.margin)
    # instantiate adapters
    pyk4a_azure = PyK4AAdapter(cfg.input_file)
    openk4a_azure = OpenK4AAdapter(cfg.input_file)
    adapters: List[AzureKinect] = [pyk4a_azure, openk4a_azure]

    # open and grab frame
    for a in adapters:
        a.open()
    capture = pyk4a_azure.read_capture()
    if capture is None:
        raise RuntimeError("Empty playback")

    depth = capture.depth
    color_py = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)

    # D2C comparison
    grid_d = sample_grid(pyk4a_azure.get_depth_size(), cfg.sample_spacing, cfg.margin)
    ref_d = pyk4a_azure.transform_d2c(grid_d, depth)
    test_d = openk4a_azure.transform_d2c(grid_d, depth)
    valid_d, errs_d = compute_comparison(ref_d, test_d)
    print(
        f"Depth -> Color: "
        f"count={len(errs_d)}, "
        f"mean={np.mean(errs_d):.5f}, "
        f"median={np.median(errs_d):.5f}, "
        f"std={np.std(errs_d):.5f}")
    draw_error_map(color_py, ref_d, valid_d, errs_d, "D2C Error Map")

    # C2D comparison
    # grid_c = sample_grid(pyk4a_azure.get_color_size(), cfg.sample_spacing, cfg.margin)
    ref_d_valid = ~np.isnan(ref_d).any(axis=1) & ~np.isnan(ref_d).any(axis=1)
    grid_c = ref_d[ref_d_valid]
    ref_c = pyk4a_azure.transform_c2d(grid_c, depth)
    test_c = openk4a_azure.transform_c2d(grid_c, depth)
    valid_c, errs_c = compute_comparison(ref_c, test_c)
    print(f"Color -> Depth: "
          f"count={len(errs_c)}, "
          f"mean={np.mean(errs_c):.5f}, "
          f"median={np.median(errs_c):.5f}, "
          f"std={np.std(errs_c):.5f}")
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    draw_error_map(depth_vis, test_c, valid_c, errs_c, "C2D Error Map")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for a in adapters:
        a.close()


if __name__ == "__main__":
    main()
