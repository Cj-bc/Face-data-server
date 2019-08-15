import pytest
from unittest import mock
from timeout_decorator import timeout, TimeoutError
from FaceDataServer.faceDetection import (_isFaceExist, _getBiggestFace
                                         , _normalization
                                         , facemark, _waitUntilFaceDetect
                                         , faceCalibration)
from conftest import (faceFrame, noFaceFrame, MockedCap)
from FaceDataServer.Types import RawFaceData, CapHasClosedError
import dlib
import numpy


# --- isFaceExist


@pytest.mark.parametrize("faceNum,expected", [(0, False), (1, True)])
def check_isFaceExist(faceNum: int, expected: bool):
    with mock.patch('FaceDataServer.faceDetection._detector',
                    return_value=dlib.rectangles(faceNum)):
        assert _isFaceExist(numpy.ndarray(0)) == expected


# --- waitUntilFaceDetect
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


# --- getBiggestFace


def test_getBiggestFace():
    emptyPoints = dlib.dpoints(194)

    biggest = dlib.dpoints(40)
    biggest.append(dlib.dpoint(100, 100))
    biggest.resize(194 + 1)
    faces = ([emptyPoints] * 5) + [biggest]

    assert _getBiggestFace(faces) == biggest


def test_getBiggestFace_noface():
    assert _getBiggestFace([]) == dlib.dpoints(194)


# --- _normalization


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


# faceCalibration


def test_faceCalibration():
    correct: RawFaceData = RawFaceData(113, 366.82966074187624
                                      , dlib.dpoint(479, 338))
    cap = MockedCap(True, faceFrame)
    with mock.patch('FaceDataServer.faceDetection.input', return_value=None):
        result: RawFaceData = faceCalibration(cap)

    assert result.eyeDistance == correct.eyeDistance
    assert result.faceHeigh == correct.faceHeigh
    assert result.faceCenter.x == correct.faceCenter.x
    assert result.faceCenter.y == correct.faceCenter.y

# -- facemark


def test_facemark():
    # definition of correct_points {{{
    correct_points = [
        (280, 295), (282, 312), (286, 329), (290, 347), (295, 364), (301, 380),
        (307, 397), (314, 413), (323, 428), (332, 442), (344, 456), (357, 468),
        (372, 478), (387, 487), (403, 494), (419, 501), (436, 508), (452, 515),
        (469, 521), (486, 526), (503, 531), (521, 534), (539, 535), (557, 534),
        (574, 530), (589, 523), (602, 512), (613, 498), (621, 483), (628, 467),
        (635, 450), (642, 433), (648, 416), (653, 399), (658, 382), (662, 365),
        (666, 348), (669, 330), (671, 313), (672, 295), (671, 277), (542, 277),
        (535, 288), (529, 301), (525, 315), (528, 329), (539, 337), (553, 341),
        (566, 342), (581, 343), (595, 344), (609, 342), (621, 336), (631, 326),
        (634, 312), (631, 298), (626, 285), (619, 272), (490, 410), (499, 405),
        (510, 401), (520, 398), (530, 394), (541, 390), (551, 387), (562, 385),
        (572, 384), (584, 383), (595, 382), (606, 381), (617, 383), (624, 390),
        (625, 400), (621, 410), (616, 419), (607, 426), (597, 431), (586, 433),
        (575, 434), (563, 433), (553, 431), (542, 429), (531, 426), (521, 423),
        (510, 420), (500, 416), (622, 398), (615, 401), (606, 403), (597, 404),
        (589, 406), (580, 407), (571, 408), (561, 408), (552, 408), (543, 408),
        (534, 408), (525, 408), (516, 409), (506, 409), (497, 408), (506, 408),
        (515, 407), (524, 407), (533, 406), (542, 404), (551, 404), (560, 404),
        (569, 404), (578, 403), (586, 402), (595, 400), (604, 398), (613, 397),
        (601, 207), (604, 201), (609, 196), (614, 192), (620, 189), (626, 186),
        (633, 185), (640, 185), (647, 186), (653, 190), (657, 195), (657, 202),
        (654, 207), (648, 210), (641, 212), (635, 213), (627, 213), (620, 212),
        (614, 210), (607, 209), (494, 213), (488, 207), (480, 202), (472, 199),
        (463, 197), (454, 198), (445, 199), (437, 202), (429, 206), (422, 210),
        (414, 216), (420, 220), (428, 222), (437, 223), (446, 224), (455, 223),
        (464, 222), (473, 220), (482, 219), (491, 219), (600, 172), (602, 163),
        (608, 155), (617, 150), (626, 147), (635, 144), (645, 142), (654, 142),
        (663, 144), (670, 149), (675, 157), (678, 168), (670, 169), (662, 166),
        (653, 165), (645, 165), (635, 166), (626, 169), (617, 172), (608, 175),
        (532, 164), (518, 154), (502, 149), (484, 147), (466, 146), (448, 146),
        (430, 148), (414, 154), (398, 162), (384, 172), (372, 184), (384, 187),
        (401, 182), (418, 177), (435, 174), (454, 172), (472, 172), (490, 173),
        (507, 176), (526, 177)]
    # }}}

    correct = dlib.dpoints()
    for p in correct_points:
        correct.append(dlib.dpoint(p[0], p[1]))

    assert facemark(faceFrame) == correct


def test_facemark_noface():
    assert facemark(noFaceFrame) is None


# _toRelative {{{
def test__toRelative(a, b, c):
    def _mkp(t: Tuple[float, float]) -> dlib.dpoint():
        return dlib.dpoint(t[0], t[1])

    def _pSub(t: Tuple[float, float], c: Tuple[float, float]) -> dlib.dpoint:
        return dlib.dpoint(t[0] - c[0], t[1] - c[1])

    assert _toRelative(dlib.dpoints([_mkp(a), _mkp(b), _mkp(c)])) \
               == dlib.dpoints([_pSub(a, c), _pSub(b, c)])
# }}}
