import pytest
import dlib

from FaceDataServer.Types import (RawFaceData, FaceRotations)
from conftest import (points_front, points_right, points_left
                     , points_upside, points_bottom
                     , points_lean_left, points_lean_right)


# RawFaceData {{{
@pytest.mark.parametrize('eD,fH,fC', [(2, 2, dlib.dpoint(0, 0)),
                                      (1, 1, dlib.dpoint(0, 0)),
                                      (1, 1, dlib.dpoint(-1, -1))
                                      ])
def test_RawFaceData_thresholded_noaffect(eD, fH, fC):
    target = RawFaceData(1, 1, dlib.dpoint(0, 0))
    assert target.thresholded(RawFaceData(eD, fH, fC)) == target


def test_RawFaceData_thresholded_affect():
    target = RawFaceData(1, 1, dlib.dpoint(0, 0))
    assert target.thresholded(RawFaceData(0, 0, dlib.dpoint(1, 1))) == \
            RawFaceData(0, 0, target.faceCenter)


# --- RawFaceData.get


def test_RawFaceData_get():
    correctRawFaceData = RawFaceData(6, 100, dlib.dpoint(0, 0))

    # faceCenter can't be compared(it compares instance, which always fail).
    # So I take this way
    result = RawFaceData.get(points_front)
    assert result.faceHeigh == correctRawFaceData.faceHeigh
    assert result.eyeDistance == correctRawFaceData.eyeDistance
    assert result.faceCenter.x == correctRawFaceData.faceCenter.x
    assert result.faceCenter.y == correctRawFaceData.faceCenter.y
# }}}

@pytest.mark.parametrize("points,th", [(points_front, (0, 0, 0)),
                                       (points_right, (0, -1, 0)),
                                       (points_left, (0, 1, 0)),
                                       (points_upside, (1, 0, 0)),
                                       (points_bottom, (-1, 0, 0)),
                                       (points_lean_left, (0, 0, 1)),
                                       (points_lean_right, (0, 0, -1))])
def test_FaceRotations_get(points, th):
    calib = RawFaceData(6.0, 100.0, dlib.dpoint(0, 0))

    result = FaceRotations.get(points, calib)

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
