from unittest import mock
import pytest
from main import main
from FaceDataServer.Types import (FaceDetectionError, CapHasClosedError)


@pytest.mark.parametrize("error", [FaceDetectionError, CapHasClosedError])
def test_main_FaceDetectionError(error):
    with mock.patch('main.faceCalibration', side_effect=error):
        with mock.patch('main.cv2.VideoCapture', return_value=None):
            with pytest.raises(SystemExit) as e:
                main()

            assert e.value.code == 1
