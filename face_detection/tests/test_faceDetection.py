import pytest
from unittest import mock
from faceDetection import isFaceExist, getBiggestFace
import dlib
import numpy


@pytest.mark.parametrize("faceNum,expected", [(0, False), (1, True)])
def test_isFaceExist(faceNum: int, expected: bool):
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
