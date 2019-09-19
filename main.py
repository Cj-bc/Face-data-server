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
from FaceDataServer.faceDataServer_pb2 import (VoidCom, FaceData, Status)
from logging import getLogger, Logger
import logging.config as loggingConfig

# Loggers {{{
configuu = { "version": 1
           , "handlers": {"console": {"class": "logging.StreamHandler"}
                        , "file": {"class": "logging.FileHandler"
                            , "filename": "faceDataServer.log"
                            , "formatter": "simpleFormatter"}}
           , "loggers": {"Servicer": {}}
           , "root": {"level": "DEBUG"
                     , "handlers": ["file"]
                     , "formatters": ["simpleFormatter"]}
           , "formatters": {"simpleFormatter": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}}
           }
loggingConfig.dictConfig(configuu)
logger: Logger = getLogger('main')
logger_servicer: Logger = getLogger('Servicer')
# }}}


# FaceDataStore {{{
class FaceDataStore():
    """ Stores current FaceData
    """
    current: FaceData = FaceData(x=0.0, y=0.0, z=0.0)
    cap: cv2.VideoCapture = None
    calib: RawFaceData = RawFaceData.default()

    def __init__(self, _cap, _calib) -> None:
        self.cap = _cap
        self.calib = _calib

    def genData(self) -> None:
        """ generate FaceData from caps until the camera is closed.
        """
        while self.cap.isOpened() is True:
            rots: FaceRotations = FaceRotations(0, 0, 0)
            face: Face = Face.default()
            _, frame = self.cap.read()
            landmark: Optional[dlib.points] = facemark(frame)

            if landmark is not None:
                face: Face = Face.fromDPoints(landmark)
                rots: FaceRotations = FaceRotations.get(face, self.calib)

            self.current = FaceData(x=rots.x, y=rots.y, z=rots.z)
# }}}


# Servicer {{{
class Servicer(grpc_faceDataServer.FaceDataServerServicer):
    """
        args:
            do_stream: Set False when stream is closed
            initialized: True after once self.init() is called
            dataStore: FaceDataStore object that holds current faceData
            dataStoreExecuter: an Executor object that holds threads used to run FaceDataStore.genData()
    """
    do_streams: int = 0
    initialized: bool = False

    dataStore: FaceDataStore = None
    dataStoreExecuter: futures.Executor = None

    def __init__(self, exe):
        self.dataStoreExecuter = exe

    def init(self, req, context):
        if self.initialized:
            return Status(success=True)

        cap: cv2.VideoCapture = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return Status(success=False,
                              exitCode=ExitCode.CameraNotFound
                                       | ExitCode.FILE_MAIN)

        try:
            calibrated: RawFaceData = faceCalibration(cap)
        except FaceDetectionError as e:
            cap.release()
            logger_servicer.info(f"ERROR: Unexpected things are happened: {e}")
            logger_servicer.info("Aborting")
            if hasattr(e, 'exitCode'):
                return Status(success=False, exitCode=e.exitCode)
            else:
                return Status(success=False
                                 , exitCode=ExitCode.FILE_MAIN
                                            | ExitCode.ERR_UNKNOWN
                                            | 0b00000001)
        self.dataStore = FaceDataStore(cap, calibrated)
        self.dataStoreExecuter.submit(self.dataStore.genData)
        self.initialized = True

        logger_servicer.debug("Calibrated.")
        logger_servicer.debug(f"cap: {cap}")
        return Status(success=True)

    def startStream(self, req, context):
        """Streams face data to the client
        """
        logger_servicer.info("called: startStream")
        if not self.dataStore.cap.isOpened():
            logger_servicer.info("camera isn't available")
            yield None

        self.do_streams += 1
        while 0 < self.do_streams:
            yield self.dataStore.current
        logger_servicer.debug("finished: startStream")

    def stopStream(self, req, context):
        """ stop streaming FaceData """
        logger_servicer.info("stopStream")
        self.do_streams -= 1
        self.dataStore.cap.release()
        logger_servicer.debug("Stream closed")
        return Status(success=True)

    def shutdown(self, req, context):
        """ shutdown this server
        """
        # TODO: Implement this 
        self.dataStore.cap.release()
        return Status(success=True)
# }}}


def main():
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        grpc_faceDataServer.add_FaceDataServerServicer_to_server(
                Servicer(executor), server)
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
