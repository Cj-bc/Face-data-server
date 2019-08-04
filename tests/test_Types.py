import pytest
import dlib

from FaceDataServer.Types import (RawFaceData)
from conftest import points_front


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
