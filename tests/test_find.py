import pytest
import dlib
from conftest import (points_front, points_right, points_left, points_upside,
                      points_bottom, points_lean_left, points_lean_right)
from FaceDataServer.Types import (RawFaceData)
from FaceDataServer.find import (rotates)


@pytest.mark.parametrize("points,th", [(points_front, (0, 0, 0)),
                                       (points_right, (0, -1, 0)),
                                       (points_left, (0, 1, 0)),
                                       (points_upside, (1, 0, 0)),
                                       (points_bottom, (-1, 0, 0)),
                                       (points_lean_left, (0, 0, 1)),
                                       (points_lean_right, (0, 0, -1))])
def test_rotates(points, th):
    calib = RawFaceData(6.0, 100.0, dlib.dpoint(0, 0))

    result = rotates(points, calib)

    def _assert(n, threshold):
        if threshold == -1:
            assert n < 0
        elif threshold == 0:
            assert n == 0
        else:
            assert n > 0

    _assert(result.x, th[0])
    _assert(result.y, th[1])
    _assert(result.z, th[2])
