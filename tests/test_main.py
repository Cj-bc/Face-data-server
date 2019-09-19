from unittest import mock
from timeout_decorator import timeout, TimeoutError
import pytest
import re
from conftest import (faceFrame, noFaceFrame)
import dlib
from main import main, faceDetectionLoop
from FaceDataServer.Types import (FaceDetectionError, CapHasClosedError
                                 , Face, RawFaceData)







# }}}








