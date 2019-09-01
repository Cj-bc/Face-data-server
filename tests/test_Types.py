from typing import Tuple
import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
import dlib
import math

from FaceDataServer.Types import (RawFaceData, FaceRotations, Part, Coord
                                 , AbsoluteCoord, RelativeCoord
                                 , Face, Eye, Mouth, Nose, EyeBrow)
from conftest import (face_front, face_right, face_left
                     , face_upside, face_bottom
                     , face_lean_left, face_lean_right
                     , finiteFloatCallable, PartStrategies, CoordStrategies
                     , AbsoluteCoordStrategies, RelativeCoordStrategies
                     , FaceStrategies
                     , round_Part, round_Coord, round_Face)


# Coord {{{
@given(finiteFloatCallable, finiteFloatCallable)
def test_Coord__eq__(n, m):
    assert Coord(n, m) == Coord(n, m)
    assert Coord(n, m) != Coord(n - 1.0, m)
    assert Coord(n, m) != Coord(n, m - 1.0)
    if n != m:
        assert Coord(n, m) != Coord(m, n)


@given(st.builds(Coord, finiteFloatCallable, finiteFloatCallable))
def test_Coord__neg__(c):
    assert -(-c) == c


@given(CoordStrategies, CoordStrategies, CoordStrategies)
def test_Coord__add__(a, b, c):
    # Needed to avoid test failures due to errors in floating point calc
    assume((a.x + b.x) + c.x == a.x + (b.x + c.x))
    assume((a.y + b.y) + c.y == a.y + (b.y + c.y))

    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + Coord(0.0, 0.0) == a
    assert a + -a == Coord(0.0, 0.0)


# TODO: Is this appropriate hypothesis?
@given(st.builds(Coord, finiteFloatCallable, finiteFloatCallable))
def test_Coord__sub__(a):
    assert a - a == Coord(0.0, 0.0)


@given(CoordStrategies, finiteFloatCallable)
def test_Coord_mul_and_div(c, d):
    assume(d != 0.0)
    assert round_Coord(c * d / d) == round_Coord(c)


def test_Coord_default():
    assert Coord.default() == Coord(0, 0)


@given(st.builds(dlib.dpoint, finiteFloatCallable, finiteFloatCallable))
def test_Coord_fromDPoint(p):
    assert Coord.fromDPoint(p) == Coord(p.x, p.y)
# }}}


# AbsoluteCoord {{{
def test_AbsoluteCoord_fromCoord():
    assert isinstance(AbsoluteCoord.fromCoord(Coord(0, 0)), AbsoluteCoord)


@given(AbsoluteCoordStrategies, AbsoluteCoordStrategies
      , AbsoluteCoordStrategies)
def test_AbsoluteCoord__add__(a, b, c):
    # Needed to avoid test failures due to errors in floating point calc
    assume((a.x + b.x) + c.x == a.x + (b.x + c.x))
    assume((a.y + b.y) + c.y == a.y + (b.y + c.y))

    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + AbsoluteCoord(0.0, 0.0) == a
    assert a + -a == AbsoluteCoord(0.0, 0.0)


@given(CoordStrategies, finiteFloatCallable)
def test_AbsoluteCoord_mul_and_div(c, d):
    assume(d != 0.0)
    assert c * d / d
# }}}


# RelativeCoord {{{
def test_RelativeCoord_fromCoord():
    assert isinstance(RelativeCoord.fromCoord(Coord(0, 0)), RelativeCoord)


@given(RelativeCoordStrategies, RelativeCoordStrategies
      , RelativeCoordStrategies)
def test_RelativeCoord__add__(a, b, c):
    # Needed to avoid test failures due to errors in floating point calc
    assume((a.x + b.x) + c.x == a.x + (b.x + c.x))
    assume((a.y + b.y) + c.y == a.y + (b.y + c.y))

    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + RelativeCoord(0.0, 0.0) == a
    assert a + -a == RelativeCoord(0.0, 0.0)


@given(CoordStrategies, finiteFloatCallable)
def test_RelativeCoord_mul_and_div(c, d):
    assume(d != 0.0)
    assert round_Coord(c * d / d) == round_Coord(c)
# }}}


# Part {{{
@given(st.tuples(finiteFloatCallable, finiteFloatCallable)
      , st.tuples(finiteFloatCallable, finiteFloatCallable)
      , st.tuples(finiteFloatCallable, finiteFloatCallable)
      , st.tuples(finiteFloatCallable, finiteFloatCallable))
def test_Part__init__(b, t, l, r):
    """ Assert both 'Coord' and 'dlib.dpoint' can be used """
    def _2C(t: Tuple[float, float]):
        return Coord(t[0], t[1])

    def _2P(t: Tuple[float, float]):
        return dlib.dpoint(t[0], t[1])

    assert Part(_2C(b), _2C(t), _2C(l), _2C(r)) \
               == Part(_2P(b), _2P(t), _2P(l), _2P(r))


def test_Part__init__typeMismatch():
    with pytest.raises(TypeError):
        Part(0, 1, 2, 3, 4)


@given(PartStrategies)
def test_Part__neg__(a):
    assert -(-a) == a


@given(PartStrategies, PartStrategies, PartStrategies)
def test_Part__add__(a, b, c):
    """ the properties are taken from here:
        http://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#t:Num
    """
    assert round_Part((a + b) + c) == round_Part(a + (b + c))
    assert a + b == b + a
    assert a + Part.default() == a
    assert a + -a == Part(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))


@given(PartStrategies)
def test_Part__sub__(a):
    assert a - a == Part(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))


@given(PartStrategies, finiteFloatCallable)
def test_Part_mul_and_div(p, d):
    assume(d != 0)

    assert round_Part(p * d / d) == round_Part(p)


def test_Part_default():
    assert Part.default()\
        == Part(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
# }}}


# Nose {{{
@given(CoordStrategies, CoordStrategies, CoordStrategies)
def test_Nose__init__(a, b, c):
    nose = Nose(a, b, c)

    assert nose.bottom == a
    assert nose.top == Coord(0, 0)
    assert nose.leftSide == b
    assert nose.rightSide == c


def test_Nose_default():
    assert Nose.default() == Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
# }}}


# Face {{{
def test_Face_default():
    assert Face.default() == Face(AbsoluteCoord.default()
                                 , RelativeCoord.default()
                                 , RelativeCoord.default()
                                 , RelativeCoord.default()
                                 , Eye.default(), Eye.default()
                                 , Mouth.default() , Nose.default()
                                 , EyeBrow.default(), EyeBrow.default())


def test_Face_fromDPoints():
    points = dlib.dpoints([dlib.dpoint(x, x) for x in range(194)])
    # Those values are defined in LANDMARK_NUM
    correct = Face(AbsoluteCoord(49.0, 49.0), RelativeCoord(0.0, 0.0)
                  , RelativeCoord(40.0, 40.0), RelativeCoord(19.0, 19.0)
                  , Eye(Coord(129.0, 129.0), Coord(120.0, 120.0)
                       , Coord(124.0, 124.0) , Coord(114.0, 114.0))
                  , Eye(Coord(149.0, 149.0), Coord(140.0, 140.0)
                       , Coord(135.0, 135.0) , Coord(145.0, 145.0))
                  , Mouth(Coord(79.0, 79.0), Coord(65.0, 65.0)
                         , Coord(71.0, 71.0) , Coord(58.0, 58.0))
                  , Nose(Coord(49.0, 49.0), Coord(54.0, 54.0)
                        , Coord(44.0, 44.0))
                  , EyeBrow(Coord(169.0, 169.0), Coord(159.0, 159.0)
                           , Coord(164.0, 164.0) , Coord(154.0, 154.0))
                  , EyeBrow(Coord(190.0, 190.0), Coord(179.0, 179.0)
                           , Coord(174.0, 174.0) , Coord(185.0, 185.0)))
    assert Face.fromDPoints(points) == correct


# is this good test?
@given(FaceStrategies, finiteFloatCallable)
def test_Face_mul_and_div(f, d):
    assume(d != 0.0)
    assume(1 / d * d == 1)
    assert round_Face(f * d / d) == round_Face(f)
# }}}


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


@pytest.mark.parametrize('face, eD, fH, fC'
                        , [(face_front, 6, 100, dlib.dpoint(0, 0))
                        , (face_left, 5, 100, dlib.dpoint(3, 0))
                        , (face_right, 5, 100, dlib.dpoint(-3, 0))
                        , (face_upside, 6, 90, dlib.dpoint(0, 5))
                        , (face_bottom, 6, 90, dlib.dpoint(0, -10))
                        , (face_lean_left, 6, 100, dlib.dpoint(0, 0))
                        , (face_lean_right, 6, 100, dlib.dpoint(0, 0))])
def test_RawFaceData_get(face, eD, fH, fC):
    correctRawFaceData = RawFaceData(eD, fH, fC)

    # faceCenter can't be compared(it compares instance, which always fail).
    # So I take this way
    result = RawFaceData.get(face)
    assert math.isclose(result.faceHeigh, correctRawFaceData.faceHeigh)\
            , f"result.faceHeigh: [{result.faceHeigh}],"\
            " cerrect: [{correctRawFaceData.faceHeigh}]"
    assert math.isclose(result.eyeDistance, correctRawFaceData.eyeDistance)\
            , f"result.eyeDistance: [{result.eyeDistance}],"\
            " correctRawFaceData.eyeDistance:"\
            " [{correctRawFaceData.eyeDistance}]"
    assert math.isclose(result.faceCenter.x, correctRawFaceData.faceCenter.x)\
            , f"result.faceCenter.x: [{result.faceCenter.x}],"\
            " correctRawFaceData.faceCenter.x:"\
            " [{correctRawFaceData.faceCenter.x}]"
    assert math.isclose(result.faceCenter.y, correctRawFaceData.faceCenter.y)\
            , f"result.faceCenter.y: [{result.faceCenter.y}],"\
            " correctRawFaceData.faceCenter.y:"\
            " [{correctRawFaceData.faceCenter.y}]"
# }}}


@pytest.mark.parametrize("points,th", [(face_front, (0, 0, 0)),
                                       (face_right, (0, -1, 0)),
                                       (face_left, (0, 1, 0)),
                                       (face_upside, (1, 0, 0)),
                                       (face_bottom, (-1, 0, 0)),
                                       (face_lean_left, (0, 0, 1)),
                                       (face_lean_right, (0, 0, -1))])
def test_FaceRotations_get(points, th):
    calib = RawFaceData(6.0, 100.0, dlib.dpoint(0, 0))

    result = FaceRotations.get(points, calib)

    def _assert(n, threshold):
        if threshold == -1:
            assert n < 0
        elif threshold == 0:
            assert math.isclose(n, 0)
        else:
            assert n > 0

    _assert(result.x, th[0])
    _assert(result.y, th[1])
    _assert(result.z, th[2])
