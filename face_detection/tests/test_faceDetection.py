import pytest
from unittest import mock
from faceDetection import isFaceExist, getBiggestFace, getRawFaceData, _normalization
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


def test_normalization():
    inList = [0, 1, 10, 86, 87, 88, 89, 90, 91, 92,
              93, 94, 95, 11, 96, 97, 98, 99, 114, 115,
              116, 117, 118, 119, 12, 120, 121, 122, 123, 124,
              125, 126, 127, 128, 129, 13, 130, 131, 132, 133,
              134, 135, 136, 137, 138, 139, 14, 140, 141, 142,
              143, 144, 145, 146, 147, 148, 149, 15, 150, 151,
              152, 153, 154, 155, 156, 157, 158, 159, 16, 160,
              161, 162, 163, 164, 165, 166, 167, 168, 169, 17,
              170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
              18, 180, 181, 182, 183, 184, 185, 186, 187, 188,
              189, 19, 190, 191, 192, 193, 2, 20, 21, 22,
              23, 24, 25, 26, 27, 28, 29, 3, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 4, 40,
              41, 42, 43, 44, 45, 46, 47, 48, 49, 5,
              50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
              6, 60, 61, 62, 63, 64, 65, 66, 67, 68,
              69, 7, 70, 71, 72, 73, 74, 75, 76, 77,
              78, 79, 8, 80, 81, 82, 83, 84, 85, 100,
              101, 102, 103, 9, 104, 105, 106, 107, 108, 109,
              110 , 111 , 112 , 113]

    assert _normalization(inList) == list(range(0, 194))
