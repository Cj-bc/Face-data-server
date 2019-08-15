from typing import Tuple
import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
import dlib

from FaceDataServer.Types import (RawFaceData, FaceRotations, Part, Coord, Nose
                                 , AbsoluteCoord, RelativeCoord)
from conftest import (points_front, points_right, points_left
                     , points_upside, points_bottom
                     , points_lean_left, points_lean_right
                     , finiteFloatCallable, PartStrategies, CoordStrategies
                     , AbsoluteCoordStrategies, RelativeCoordStrategies)


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


# TODO: Is this appropriate hypothesis?
@given(finiteFloatCallable, finiteFloatCallable, finiteFloatCallable)
def test_Coord__truediv__(x, y, d):
    assume(d != 0.0)
    assert Coord(x, y) / d == Coord(x / d, y / d)


def test_Coord_default():
    assert Coord.default() == Coord(0, 0)


@given(st.builds(dlib.dpoint, finiteFloatCallable, finiteFloatCallable))
def test_Coord_fromDPoint(p):
    assert Coord.fromDPoint(p) == Coord(p.x, p.y)
# }}}


# AbsoluteCoord {{{
def test_AbsoluteCoord_fromCoord():
    assert type(AbsoluteCoord.fromCoord(Coord(0, 0))) == type(AbsoluteCoord(0, 0))


@given(AbsoluteCoordStrategies, AbsoluteCoordStrategies, AbsoluteCoordStrategies)
def test_AbsoluteCoord__add__(a, b, c):
    # Needed to avoid test failures due to errors in floating point calc
    assume((a.x + b.x) + c.x == a.x + (b.x + c.x))
    assume((a.y + b.y) + c.y == a.y + (b.y + c.y))

    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + AbsoluteCoord(0.0, 0.0) == a
    assert a + -a == AbsoluteCoord(0.0, 0.0)


# TODO: Is this appropriate hypothesis?
@given(finiteFloatCallable, finiteFloatCallable, finiteFloatCallable)
def test_AbsoluteCoord__truediv__(x, y, d):
    assume(d != 0.0)
    assert AbsoluteCoord(x, y) / d == AbsoluteCoord(x / d, y / d)
# }}}


# RelativeCoord {{{
def test_RelativeCoord_fromCoord():
    assert type(RelativeCoord.fromCoord(Coord(0, 0))) == type(RelativeCoord(0, 0))


@given(RelativeCoordStrategies, RelativeCoordStrategies, RelativeCoordStrategies)
def test_RelativeCoord__add__(a, b, c):
    # Needed to avoid test failures due to errors in floating point calc
    assume((a.x + b.x) + c.x == a.x + (b.x + c.x))
    assume((a.y + b.y) + c.y == a.y + (b.y + c.y))

    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + RelativeCoord(0.0, 0.0) == a
    assert a + -a == RelativeCoord(0.0, 0.0)


# TODO: Is this appropriate hypothesis?
@given(finiteFloatCallable, finiteFloatCallable, finiteFloatCallable)
def test_RelativeCoord__truediv__(x, y, d):
    assume(d != 0.0)
    assert RelativeCoord(x, y) / d == RelativeCoord(x / d, y / d)
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

    assert Part(_2C(b), _2C(t), _2C(l), _2C(r)) == \
                Part(_2P(b), _2P(t), _2P(l), _2P(r))


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
    assert (a + b) + c == a + (b + c)
    assert a + b == b + a
    assert a + Part.default() == a
    assert a + -a == Part(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))


@given(PartStrategies)
def test_Part__sub__(a):
    assert a - a == Part(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))


@given(PartStrategies, finiteFloatCallable)
def test_Part__truediv__(p, d):
    assume(d != 0)

    assert p / d == Part(p.bottom / d, p.top / d
                        , p.leftSide / d, p.rightSide / d)


def test_Part_default():
    assert Part.default() == Part(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
# }}}


# Nose {{{
@given(CoordStrategies, CoordStrategies, CoordStrategies)
def test_Nose__init__(a, b, c):
    nose = Nose(a, b, c)

    assert nose.bottom == a
    assert nose.top == None
    assert nose.leftSide == b
    assert nose.rightSide == c


def test_Nose_default():
    assert Nose.default() == Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
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
