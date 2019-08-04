# THIS CODE IS NOT WRITTEN BY ME, BUT BY @kekeho
# Refer to: https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
#
import dlib
import math
from .Types import RawFaceData, FaceRotations, LANDMARK_NUM



# rotates(landmark: dlib.dpoints, calib: RawFaceData) -> FaceRotations
def rotates(landmark: dlib.dpoints, calib: RawFaceData) -> FaceRotations:
    """ calculate face rotations from calibration data and landmark
    """
    eyeLineVector = landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]] - \
                            landmark[LANDMARK_NUM["LEFT_EYE_BOTTOM"]]
    raw = RawFaceData.get(landmark).thresholded(calib)

    # TODO: how can I notice which side does face face to?
    #       I can't simply compare eyes sizes, 'cus sometimes
    #       user might wink. In that case, I can't recognize properly.
    degreeY = math.acos(raw.eyeDistance / calib.eyeDistance)
    degreeX = math.acos(raw.faceHeigh / calib.faceHeigh)
    degreeZ = math.atan(eyeLineVector.y / eyeLineVector.x)
    # TODO: ^ This some times got error 'Division by 0'

    rotateX = degreeX if raw.faceCenter.y > calib.faceCenter.y\
                        else -1 * degreeX
    rotateY = degreeY if raw.faceCenter.x > calib.faceCenter.x\
                        else -1 * degreeY
    # v Is this correct code? v
    rotateZ = degreeZ
    return FaceRotations(rotateX, rotateY, rotateZ)
