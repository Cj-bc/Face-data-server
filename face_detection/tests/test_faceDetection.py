import pytest
from unittest import mock
from faceDetection import isFaceExist
import dlib
import numpy


@pytest.mark.parametrize("faceNum,expected", [(0, False), (1, True)])
def test_isFaceExist(faceNum, expected):
    with mock.patch('faceDetection.detector', return_value=dlib.rectangles(faceNum)):
        assert isFaceExist(numpy.ndarray(0)) == expected
