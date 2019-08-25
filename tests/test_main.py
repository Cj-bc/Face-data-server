from unittest import mock
from timeout_decorator import timeout, TimeoutError
import pytest
import re
from conftest import (faceFrame, noFaceFrame)
import dlib
import cv2
from main import main, faceDetectionLoop
from FaceDataServer.Types import (FaceDetectionError, CapHasClosedError, Face, RawFaceData)


# faceDetectionLoop {{{
def test_faceDetectionLoop_closedCap():
    """ Don't remain recursive call if cap is closed.
    """
    class mockedClosedCap:
        def isOpened(self) -> bool:
            return False

    def_cap = mockedClosedCap()
    def_RawFaceData = RawFaceData(0.0, 0.0, dlib.dpoint(0, 0))
    def_Face = Face.default()
    ret_cap, ret_calib, ret_prevFace = faceDetectionLoop(def_cap, def_RawFaceData, def_Face)
    assert def_cap == ret_cap
    assert def_RawFaceData == ret_calib
    assert def_Face == ret_prevFace


def test_faceDetectionLoop_WithFace():
    """ test faceDetectionLoop remains outputting values
    while face is recognized. """
    class mockedCap:
        def isOpened(self) -> bool:
            return True

        def read(self):
            return (True, faceFrame)

    def_cap = mockedCap()
    def_RawFaceData = RawFaceData(6, 100.0, dlib.dpoint(0, 0))
    def_Face = Face.default()

    @timeout(10)
    def _run():
        faceDetectionLoop(def_cap, def_RawFaceData, def_Face)

    with pytest.raises(TimeoutError):
        _run()


def test_faceDetectionLoop_withoutFace(capsys):
    """ test faceDetectionLoop remains outputting default values
    while face is not recognized. """
    class mockedCap:
        def isOpened(self) -> bool:
            return True

        def read(self):
            return (False, noFaceFrame)

    def_cap = mockedCap()
    def_RawFaceData = RawFaceData(6, 100.0, dlib.dpoint(0, 0))
    def_Face = Face.default()

    @timeout(10)
    def _run():
        faceDetectionLoop(def_cap, def_RawFaceData, def_Face)

    with pytest.raises(TimeoutError):
        _run()

    stdout = capsys.readouterr()
    assert re.match(r'(\[[^]]*] 0.0, 0.0, 0.0\n)*', stdout.out) is not None
# }}}


@pytest.mark.parametrize("error", [FaceDetectionError, CapHasClosedError])
def test_main_FaceDetectionError(error):
    with mock.patch('main.faceCalibration', side_effect=error):
        with mock.patch('main.cv2.VideoCapture', return_value=None):
            with pytest.raises(SystemExit) as e:
                main()

            assert e.value.code == 1


def test_main_NoCameraFound(capsys):
    class MockedCap():
        def isOpened(self):
            return False

    with mock.patch('main.cv2.VideoCapture', return_value=MockedCap()):
        with pytest.raises(SystemExit) as e:
            main()

        stdout = capsys.readouterr()
        assert stdout.out == "connecting to camera...\nERROR: Cannot connect to camera\n Aborting\n"
        assert e.value.code == 1
