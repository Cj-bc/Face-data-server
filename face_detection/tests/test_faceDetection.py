import pytest
from unittest import mock
from faceDetection import isFaceExist, getBiggestFace, getRawFaceData
from Types import RawFaceData
from typing import List, Tuple
import dlib
import numpy
from functools import reduce


def test_getRawFaceData():
    faceRawPoints: List[Tuple[int, int]] = [
                     (-50, 25), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, -50),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (50, 25), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (15, 40),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (-5, 40), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 50),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (25, 50),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 50),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                     (0, 0), (0, 0), (0, 0)]
    # # for DEUBG. Use this to check faceRawPoints are correctly set.
    # for v in list(LANDMARK_NUM.items()):
    #     print(f"{v[0]}: {faceRawPoints[v[1]]}")

    face = dlib.points(0)
    facePoints: List[dlib.point] = map(lambda t: dlib.point(t[0], t[1]),
                                       faceRawPoints)
    face: dlib.points = dlib.points()
    for p in facePoints:
        face.append(p)

    correctRawFaceData = RawFaceData(20, 100, dlib.point(0, 0))

    # faceCenter can't be compared(it compares instance, which always fail).
    # So I take this way
    result = getRawFaceData(face)
    assert result.faceHeigh == correctRawFaceData.faceHeigh
    assert result.eyeDistance == correctRawFaceData.eyeDistance
    assert result.faceCenter.x == correctRawFaceData.faceCenter.x
    assert result.faceCenter.y == correctRawFaceData.faceCenter.y


@pytest.mark.parametrize("faceNum,expected", [(0, False), (1, True)])
def check_isFaceExist(faceNum: int, expected: bool):
    with mock.patch('faceDetection.detector',
                    return_value=dlib.rectangles(faceNum)):
        assert isFaceExist(numpy.ndarray(0)) == expected


def test_getBiggestFace():
    emptyPoints = dlib.points(41)

    biggest = dlib.points(40)
    biggest.append(dlib.point(100, 100))
    faces = ([emptyPoints] * 5) + [biggest]

    assert getBiggestFace(faces) == biggest


def test_getBiggestFace_noface():
    assert getBiggestFace([]) == dlib.points(41)
