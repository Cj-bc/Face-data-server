from unittest import mock
import pytest
import dlib
from conftest import (points_front, points_right, points_left, points_upside,
                      points_bottom, points_lean_left, points_lean_right)
from main import main
from FaceDataServer.Types import (RawFaceData,
                      FaceDetectionError, CapHasClosedError)
from FaceDataServer.find import (point_abs, area_rect, rotates)


@pytest.mark.parametrize("x,y", [(-1, -1), (-1, 1), (1, -1),
                                 (0, 0), (1, 0), (0, 1), (1, 1),
                                 (-1, 0), (0, -1)])
def test_point_abs(x: int, y: int):
    result = point_abs(dlib.dpoint(x, y))
    assert result.x == abs(x)
    assert result.y == abs(y)


@pytest.mark.parametrize("a,b,c,d,correct",
                         [((0, 0), (0, 0), (0, 0), (0, 0), 0),
                          ((1, 1), (-1, 1), (-1, -1), (1, -1), 4),
                          ((1, 0), (0, 1), (-1, 0), (0, -1), 4),
                          ((1, 1), (0, 0), (-1, -1), (1, 0), 4),
                          ((2, 2), (0, 0), (-2, -2), (1, 1), 16)])
def test_area_rect(a, b, c, d, correct):
    """ tested cases
        1. No area can be there
        2. Square
        3. All points are on each edge of rectangle
        4. One point is inside of area
        5. Two point is inside of area
    """
    aP = dlib.dpoint(a[0], a[1])
    bP = dlib.dpoint(b[0], b[1])
    cP = dlib.dpoint(c[0], c[1])
    dP = dlib.dpoint(d[0], d[1])
    assert area_rect(aP, bP, cP, dP) == correct


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


@pytest.mark.parametrize("error", [FaceDetectionError, CapHasClosedError])
def test_main_FaceDetectionError(error):
    with mock.patch('main.faceCalibration', side_effect=error):
        with mock.patch('main.cv2.VideoCapture', return_value=None):
            with pytest.raises(SystemExit) as e:
                main()

            assert e.value.code == 1
