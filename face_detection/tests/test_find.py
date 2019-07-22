import pytest
import dlib
from find import (point_abs
                    )


@pytest.mark.parametrize("x,y", [(-1, -1), (-1, 1), (1, -1),
                              (0, 0), (1, 0), (0, 1), (1, 1),
                              (-1, 0), (0, -1)])
def test_point_abs(x: int, y: int):
    result = point_abs(dlib.point(x, y))
    assert result.x == abs(x)
    assert result.y == abs(y)
