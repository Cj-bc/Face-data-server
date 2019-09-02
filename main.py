import cv2
import dlib
import time
import datetime
import sys
import grpc
from concurrent import futures
from typing import (Optional)

from FaceDataServer.faceDetection import (faceCalibration, facemark)
from FaceDataServer.Types import (RawFaceData, FaceRotations,
                                 FaceDetectionError, Face, ExitCode)
import FaceDataServer.faceDataServer_pb2_grpc as grpc_faceDataServer
from FaceDataServer.faceDataServer_pb2 import (ReqCommand, FaceData)


# Servicer {{{
class Servicer(grpc_faceDataServer.FaceDataServerServicer):
    do_stream: bool = False
    initialized: bool = False

    calib: RawFaceData = RawFaceData.default()
    cap: cv2.VideoCapture = None


    def init(self, req, context):
        cap: cv2.VideoCapture = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return InitStatus(success=False,
                              extCode=ExitCode.CameraNotFound | ExitCode.FILE_MAIN)

        try:
            calibrated: RawFaceData = faceCalibration(cap)
        except FaceDetectionError as e:
            cap.release()
            print(f"ERROR: Unexpected things are happened: {e}")
            print("Aborting")
            if hasattr(e, 'exitCode'):
                return InitStatus(success=False, exitCode=e.exitCode)
            else:
                return InitStatus(success=False
                                 , exitCode=ExitCode.FILE_MAIN
                                            | ExitCode.ERR_UNKNOWN
                                            | 0b00000001)
        self.calib = calibrated
        self.cap = cap
        return InitStatus(success=True)

    def startStream(self, req, context):
        """Streams face data to the client
        """
        if not cap.isOpened():
            yield None

        while self.do_stream == True:
            rots: FaceRotations = FaceRotations(0, 0, 0)
            face: Face = Face.default()
            _, frame = cap.read()
            landmark: Optional[dlib.points] = facemark(frame)

            if landmark is not None:
                face: Face = Face.fromDPoints(landmark)
                rots: FaceRotations = FaceRotations.get(face, self.calib)

            yield FaceData(x=rots.x, y=rots.y, z=rots.z)

    def stopStream(self, req, context):
        """ stop streaming FaceData """
        self.do_stream = False
        cap.release()
        return Status(success=True)

# }}}



def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_faceDataServer.add_FaceDataServerServicer_to_server(
            Servicer(), server)
    server.add_insecure_port('[::]:5039')
    server.start()
    print("server started...")
    try:
        while True:
            _ONE_DAY_IN_SECONDS = 60 * 60 * 24
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped.")


if __name__ == '__main__':
    main()
