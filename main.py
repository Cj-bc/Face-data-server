import cv2
import dlib
import datetime
import sys
from typing import (Optional)

from FaceDataServer.faceDetection import (faceCalibration, facemark)
from FaceDataServer.Types import (RawFaceData, FaceRotations,
                                 FaceDetectionError, Face)


def faceDetectionLoop(cap: cv2.VideoCapture, _calib: RawFaceData
                     , prevFace: Face):
    """ capture face image and output each rotations
        [Recursive method]
    """
    if not cap.isOpened():
        return (cap, _calib, prevFace)

    rots: FaceRotations = FaceRotations(0, 0, 0)
    face: Face = Face.default()
    _, frame = cap.read()
    landmark: Optional[dlib.points] = facemark(frame)

    if landmark is not None:
        face: Face = Face.fromDPoints(landmark)
        rots: FaceRotations = FaceRotations.get(face, _calib)

    print(f"[{datetime.datetime.today()}] {rots.x}, {rots.y}, {rots.z}")

    return faceDetectionLoop(cap, _calib, face)


def main():
    print("connecting to camera...")
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    print("camera connected.")

    try:
        calibrated: RawFaceData = faceCalibration(cap)
    except FaceDetectionError as e:
        print(f"ERROR: Unexpected things are happened: {e}")
        print("Aborting")
        sys.exit(1)

    _, _, _ = faceDetectionLoop(cap, calibrated, Face.default())

    cap.release()


if __name__ == '__main__':
    main()
