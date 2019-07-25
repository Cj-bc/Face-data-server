import cv2
import dlib
import datetime
import sys
from typing import (Optional)

from faceDetection.faceDetection import (faceCalibration, facemark)
from faceDetection.find import (rotates)
from faceDetection.Types import (RawFaceData, FaceRotations,
                                 FaceDetectionError)


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

    while cap.isOpened():
        rots: FaceRotations = FaceRotations(0, 0, 0)
        _, frame = cap.read()
        landmark: Optional[dlib.points] = facemark(frame)

        if landmark is not None:
            rots: FaceRotations = rotates(landmark, calibrated)

        print(f"{datetime.datetime.today()}: {rots.x}, {rots.y}, {rots.z}")

    cap.release()


if __name__ == '__main__':
    main()
