import numpy as np

from pose_detector import PoseDetector


def test_no_person_gives_none():
    with PoseDetector() as detector:
        assert detector.detect(np.zeros((480, 640, 3), np.uint8), 0) is None
