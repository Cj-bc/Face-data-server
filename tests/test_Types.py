import pytest
import dlib

from faceDetection.Types import (RawFaceData)


@pytest.mark.parametrize('eD,fH,fC', [(2, 2, dlib.point(0, 0)),
                                      (1, 1, dlib.point(0, 0)),
                                      (1, 1, dlib.point(-1, -1))
                                      ])
def test_RawFaceData_thresholded_noaffect(eD, fH, fC):
    target = RawFaceData(1, 1, dlib.point(0, 0))
    assert target.thresholded(RawFaceData(eD, fH, fC)) == target


def test_RawFaceData_thresholded_affect():
    target = RawFaceData(1, 1, dlib.point(0, 0))
    assert target.thresholded(RawFaceData(0, 0, dlib.point(1, 1))) == \
            RawFaceData(0, 0, target.faceCenter)
