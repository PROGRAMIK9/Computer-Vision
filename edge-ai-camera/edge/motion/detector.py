"""Lightweight frame-difference motion detector."""

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, threshold: int = 25, min_contour_area: float = 800, blur_size: int = 21):
        if blur_size < 1:
            raise ValueError("blur_size must be positive")
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        # Gaussian kernels must have odd dimensions.
        self.blur_size = blur_size if blur_size % 2 else blur_size + 1
        self._previous: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> tuple[bool, list[tuple[int, int, int, int]]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        if self._previous is None:
            self._previous = blurred
            return False, []

        difference = cv2.absdiff(self._previous, blurred)
        self._previous = blurred
        _, mask = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                x, y, width, height = cv2.boundingRect(contour)
                boxes.append((x, y, width, height))
        return bool(boxes), boxes

