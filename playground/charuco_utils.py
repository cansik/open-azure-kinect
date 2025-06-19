from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass
class MarkerDetectionResult:
    corners: Optional[np.ndarray]
    ids: Optional[np.ndarray]

    def annotate(self, image: np.ndarray):
        cv2.aruco.drawDetectedMarkers(image, self.corners, self.ids)


@dataclass
class BoardDetectionResult:
    corners: Optional[np.ndarray]
    ids: Optional[np.ndarray]

    def annotate(self, image: np.ndarray):
        cv2.aruco.drawDetectedCornersCharuco(image, self.corners, self.ids)


class CharucoDetectionHelper:
    def __init__(self, squares_x: int = 8, squares_y: int = 8,
                 square_length: float = 20, marker_length: float = 15,
                 dictionary: int = cv2.aruco.DICT_4X4_250) -> None:
        """
        Initialize the CharucoDetector with the given board parameters and dictionary.

        :param squares_x: Number of squares in the X direction (along the board's width).
        :param squares_y: Number of squares in the Y direction (along the board's height).
        :param square_length: Length of the squares in meters.
        :param marker_length: Length of the markers in meters.
        :param dictionary: ArUco dictionary to use for marker detection.
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.dictionary = dictionary

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.aruco_dict)

        self.board_detector = cv2.aruco.CharucoDetector(self.board)

        detection_params = cv2.aruco.DetectorParameters()
        self.marker_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(self.dictionary),
                                                       detection_params)

    def detect_markers(self, image: np.ndarray) -> Optional[MarkerDetectionResult]:

        gray = self._pre_process_image(image)
        corners, ids, rejected_corners = self.marker_detector.detectMarkers(gray)

        if corners is None or len(corners) == 0:
            return None

        return MarkerDetectionResult(np.array(corners), np.array(ids))

    def detect_board(self, image: np.ndarray) -> Optional[BoardDetectionResult]:
        """
        Detect the ChArUco board in the given image.

        :param image: The image in which to detect the ChArUco board.
        :return: DetectionResult containing the detected corners, ids, and the image with the detected board drawn.
        """

        gray = self._pre_process_image(image)
        corners, ids, rejected_corners, rejected_ids = self.board_detector.detectBoard(gray)

        if corners is None or len(corners) == 0:
            return None

        return BoardDetectionResult(np.array(corners), np.array(ids))

    def generate_board(self, image_size: Tuple[int, int] = (4960, 3508)) -> np.ndarray:
        """
        Draw the ChArUco board.

        :param image_size: Size of the image to draw the board in.
        :return: Image with the drawn ChArUco board.
        """
        board_image = self.board.generateImage(image_size)
        return board_image

    @staticmethod
    def _pre_process_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image


if __name__ == "__main__":
    image = cv2.imread("assets/marker.jpeg")

    detector = CharucoDetectionHelper(8, 5, 20, 15, cv2.aruco.DICT_4X4_250)
    result = detector.detect_markers(image)
    result.annotate(image)
    cv2.imwrite("assets/marker-ann.jpeg", image)

    cv2.imwrite("assets/board.png", detector.generate_board((4960, 3508)))
