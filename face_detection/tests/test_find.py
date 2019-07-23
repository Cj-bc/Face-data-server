import pytest
import dlib
from find import (point_abs, area_rect
                    )


@pytest.mark.parametrize("x,y", [(-1, -1), (-1, 1), (1, -1),
                              (0, 0), (1, 0), (0, 1), (1, 1),
                              (-1, 0), (0, -1)])
def test_point_abs(x: int, y: int):
    result = point_abs(dlib.point(x, y))
    assert result.x == abs(x)
    assert result.y == abs(y)


@pytest.mark.parametrize("a,b,c,d,correct",
                         [((0, 0), (0, 0), (0, 0), (0, 0), 0),
                          ((1, 1), (-1, 1), (-1, -1), (1, -1), 4),
                          ((1, 0), (0, 1), (-1, 0), (0, -1), 4),
                          ((1, 1), (0, 0), (-1, -1), (1, 0), 4),
                          ((2, 2), (0, 0), (-2, -2), (1, 1), 16)])
def test_area_rect(a, b, c, d, correct):
    """ tested cases
        1. No area can be there
        2. Square
        3. All points are on each edge of rectangle
        4. One point is inside of area
        5. Two point is inside of area
    """
    aP = dlib.point(a[0], a[1])
    bP = dlib.point(b[0], b[1])
    cP = dlib.point(c[0], c[1])
    dP = dlib.point(d[0], d[1])
    assert area_rect(aP, bP, cP, dP) == correct
