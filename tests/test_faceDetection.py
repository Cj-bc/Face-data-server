import pytest
from unittest import mock
from hypothesis import given
import hypothesis.strategies as st
from timeout_decorator import timeout, TimeoutError
from typing import Tuple
from FaceDataServer.faceDetection import (_isFaceExist, _getBiggestFace
                                         , _normalization
                                         , facemark, _waitUntilFaceDetect
                                         , faceCalibration, _toRelative)
from conftest import (faceFrame, noFaceFrame, MockedCap, finiteFloatCallable)
from FaceDataServer.Types import RawFaceData, CapHasClosedError
import dlib
import numpy


# isFaceExist {{{
@pytest.mark.parametrize("faceNum,expected", [(0, False), (1, True)])
def check_isFaceExist(faceNum: int, expected: bool):
    with mock.patch('FaceDataServer.faceDetection._detector',
                    return_value=dlib.rectangles(faceNum)):
        assert _isFaceExist(numpy.ndarray(0)) == expected
# }}}


# waitUntilFaceDetect {{{
@pytest.mark.parametrize("frame", [faceFrame, noFaceFrame])
def test_waitUntilFaceDetect_CapHasClosedError(frame):
    cap = MockedCap(False, frame)

    with pytest.raises(CapHasClosedError):
        _waitUntilFaceDetect(cap)


def test_waitUntilFaceDetect_faceFound():
    cap = MockedCap(True, faceFrame)

    assert _waitUntilFaceDetect(cap).all() == faceFrame.all()


def test_waitUntilFaceDetect_faceNotFound():
    cap = MockedCap(True, noFaceFrame)

    @timeout(10)
    def _run():
        _waitUntilFaceDetect(cap)

    with pytest.raises(TimeoutError):
        _run()
# }}}


# getBiggestFace {{{
def test_getBiggestFace():
    emptyPoints = dlib.dpoints(194)

    biggest = dlib.dpoints(40)
    biggest.append(dlib.dpoint(100, 100))
    biggest.resize(194 + 1)
    faces = ([emptyPoints] * 5) + [biggest]

    assert _getBiggestFace(faces) == biggest


def test_getBiggestFace_noface():
    assert _getBiggestFace([]) == dlib.dpoints(194)
# }}}


# _normalization {{{
def test_normalization():
    # inList {{{
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
    # }}}

    testPoints = dlib.dpoints(list(map(lambda n: dlib.dpoint(n, n),
                              inList)))
    correct = dlib.dpoints(list(map(lambda n: dlib.dpoint(n, n),
                                    list(range(0, 194)))))
    assert _normalization(testPoints) == correct
# }}}


# faceCalibration {{{
def test_faceCalibration():
    correct: RawFaceData = RawFaceData(113, 366.82966074187624
                                      , dlib.dpoint(0, 0))
    cap = MockedCap(True, faceFrame)
    with mock.patch('FaceDataServer.faceDetection.input', return_value=None):
        result: RawFaceData = faceCalibration(cap)

    assert result.eyeDistance == correct.eyeDistance
    assert result.faceHeigh == correct.faceHeigh
    assert result.faceCenter.x == correct.faceCenter.x
    assert result.faceCenter.y == correct.faceCenter.y
# }}}


# facemark {{{
def test_facemark():
    # definition of correct_points {{{
    correct_points = [(-301, -48), (-299, -31), (-295, -14), (-291, 4)
                     , (-286, 21), (-280, 37), (-274, 54), (-267, 70)
                     , (-258, 85), (-249, 99), (-237, 113), (-224, 125)
                     , (-209, 135), (-194, 144), (-178 , 151), (-162, 158)
                     , (-145, 165), (-129, 172), (-112, 178), (-95, 183)
                     , (-78, 188), (-60, 191), (-42, 192), (-24, 191)
                     , (-7, 187), (8, 180), (21, 169), (32, 155)
                     , (40, 140), (47, 124), (54, 107), (61, 90)
                     , (67, 73), (72, 56), (77, 39), (81, 22), (85, 5)
                     , (88, -13), (90, -30), (91, -48), (90, -66)
                     , (-39, -66), (-46, -55), (-52, -42), (-56, -28)
                     , (-53, - 14), (-42, -6), (-28, -2), (-15, -1)
                     , (0, 0), (14, 1), (28, -1), (40, -7)
                     , (50, -17), (53, -31), (50, -45), (45, -58)
                     , (38, -71), (-91, 67), (-82, 62), (-71, 58)
                     , (-61, 55), (-51, 51), (-40, 47), (-30, 44)
                     , (-19, 42), (-9, 41), (3, 40), (14, 39)
                     , (25, 38), (36, 40), (43, 47), (44, 57)
                     , (40, 67), (35, 76), (26, 83), (16, 88)
                     , (5, 90), (-6, 91), (-18, 90), (-28, 88)
                     , (-39, 86), (-50, 83), (-60, 80), (-71, 77)
                     , (-81, 73), (41, 55), (34, 58), (25, 60)
                     , (16, 61), (8, 63), (-1, 64), (-10, 65)
                     , (-20, 65), (-29, 65), (-38, 65), (-47, 65)
                     , (-56, 65), (-65, 66), (-75, 66), (-84, 65)
                     , (-75, 65), (-66, 64), (-57, 64), (-48, 63)
                     , (-39, 61), (-30, 61), (-21, 61), (-12, 61)
                     , (-3, 60), (5, 59), (14, 57), (23, 55)
                     , (32, 54), (20, -136), (23, -142), (28, -147)
                     , (33, -151), (39, -154), (45, -157), (52, -158)
                     , (59, -158), (66, -157), (72, -153), (76, -148)
                     , (76, -141), (73, -136), (67, -133), (60, -131)
                     , (54, -130), (46, -130), (39, -131), (33, -133)
                     , (26, -134), (-87, -130), (-93, -136), (-101, -141)
                     , (-109, -144), (-118, -146), (-127, -145), (-136, -144)
                     , (-144, -141), (-152, -137), (-159, -133), (-167, -127)
                     , (-161, -123), (-153, -121), (-144 , -120), (-135, -119)
                     , (-126, -120), (-117, -121), (-108, -123), (-99, -124)
                     , (-90, -124), (19, -171), (21, -180), (27, -188)
                     , (36, -193), (45, -196) , (54, -199), (64, -201)
                     , (73, -201), (82, -199), (89, -194), (94, -186)
                     , (97, -175), (89, -174), (81, -177), (72, -178)
                     , (64, -178), (54, -177), (45, -174), (36, -171)
                     , (27, -168), (-49, -179), (-63, -189), (-79, -194)
                     , (-97, -196), (-115, -197), (-133, -197), (-151, -195)
                     , (-167, -189), (-183, -181), (-197, -171), (-209, -159)
                     , (-197, -156), (-180, -161), (-163, -166), (-146, -169)
                     , (-127, -171), (-109, -171), (-91, -170), (-74, -167)
                     , (-55, -166)]
# }}}

    correct = dlib.dpoints()
    for p in correct_points:
        correct.append(dlib.dpoint(p[0], p[1]))

    assert facemark(faceFrame) == correct


def test_facemark_noface():
    assert facemark(noFaceFrame) is None
# }}}


# _toRelative {{{
@given(st.tuples(finiteFloatCallable, finiteFloatCallable)
      , st.tuples(finiteFloatCallable, finiteFloatCallable)
      , st.tuples(finiteFloatCallable, finiteFloatCallable))
def test__toRelative(a, b, c):
    def _mkp(t: Tuple[float, float]) -> dlib.dpoint:
        return dlib.dpoint(t[0], t[1])

    def _pSub(t: Tuple[float, float], c: Tuple[float, float]) -> dlib.dpoint:
        return dlib.dpoint(t[0] - c[0], t[1] - c[1])

    assert _toRelative(dlib.dpoints([_mkp(a), _mkp(b)]), _mkp(c)) \
               == dlib.dpoints([_pSub(a, c), _pSub(b, c)])
# }}}
