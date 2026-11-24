import numpy as np
import pytest

from app import MAX_SIZE, Recorder, knee_angle, resize

FRAME = np.zeros((10, 10, 3), np.uint8)


def make_points():
    """33 de puncte, cu genunchiul stang indoit la 90 de grade si cel drept intins."""
    points = np.ones((33, 3))
    points[[23, 25, 27], :2] = [(100, 0), (0, 0), (0, 100)]
    points[[24, 26, 28], :2] = [(0, 0), (0, 100), (0, 200)]
    return points


def test_knee_angle_uses_the_right_landmarks():
    assert knee_angle(make_points(), "left") == pytest.approx(90)
    assert knee_angle(make_points(), "right") == pytest.approx(180)


def test_hidden_landmark_gives_no_angle():
    points = make_points()
    points[27, 2] = 0.2  # glezna stanga acoperita
    assert knee_angle(points, "left") is None
    assert knee_angle(points, "right") is not None


def test_recorder_picks_the_more_visible_leg():
    points = make_points()
    points[[23, 25, 27], 2] = 0.6
    recorder = Recorder()
    recorder.add(0.0, FRAME, points)
    assert recorder.side == "right"


def test_recorder_handles_frames_without_person(tmp_path):
    recorder = Recorder()
    recorder.add(0.0, FRAME, None)
    assert recorder.angles["left"] == [None]
    assert recorder.save(tmp_path) is False


def test_recorder_saves_all_figures(tmp_path):
    recorder = Recorder()
    recorder.add(0.0, FRAME, make_points())
    recorder.add(0.1, FRAME, None)
    assert recorder.save(tmp_path) is True
    assert {p.name for p in tmp_path.iterdir()} == {
        "1_sursa.png", "2_puncte.png", "3a_jos.png", "3b_sus.png", "4_grafic_unghi.png", "unghiuri.csv",
    }


def test_resize_keeps_aspect_ratio_and_never_enlarges():
    assert resize(np.zeros((1080, 1920, 3), np.uint8)).shape == (540, MAX_SIZE, 3)
    assert resize(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)
