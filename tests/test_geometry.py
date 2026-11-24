import math

import pytest

from geometry import angle


# Valori calculate manual; toleranta de 1 grad vine din obiectivul O2
@pytest.mark.parametrize("a, b, c, expected", [
    ((1, 0), (0, 0), (2, 0), 0),
    ((1, 0), (0, 0), (1, 1), 45),
    ((1, 0), (0, 0), (1, math.sqrt(3)), 60),
    ((1, 0), (0, 0), (0, 1), 90),
    ((1, 0), (0, 0), (-1, 1), 135),
    ((0, 0), (0, 5), (0, 10), 180),
])
def test_known_angles(a, b, c, expected):
    assert angle(a, b, c) == pytest.approx(expected, abs=1)


def test_does_not_depend_on_order_position_or_scale():
    reference = angle((1, 0), (0, 0), (1, 2))
    assert angle((1, 2), (0, 0), (1, 0)) == pytest.approx(reference)
    assert angle((110, 100), (100, 100), (110, 120)) == pytest.approx(reference)


def test_coincident_points_give_nan():
    assert math.isnan(angle((3, 3), (3, 3), (5, 5)))
