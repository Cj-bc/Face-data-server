import cv2
import dlib
from contextlib import closing
import time
import secrets
import grpc
import socket
from concurrent import futures
from typing import (Optional, List)

from FaceDataServer.faceDetection import (faceCalibration, facemark)
from FaceDataServer.Types import (RawFaceData, FaceRotations,
                                 FaceDetectionError, Face, ExitCode)
import FaceDataServer.faceDataServer_pb2_grpc as grpc_faceDataServer
from FaceDataServer.faceDataServer_pb2 import (FaceData, Status, Token)
from logging import getLogger, Logger
import logging.config as loggingConfig

# Loggers {{{
configuu = {"version": 1
           , "handlers": {"console": {"class": "logging.StreamHandler"}
                        , "file": {"class": "logging.FileHandler"
                            , "filename": "faceDataServer.log"
                            , "formatter": "simpleFormatter"}}
           , "loggers": {"Servicer": {}}
           , "root": {"level": "DEBUG"
                     , "handlers": ["file"]
                     , "formatters": ["simpleFormatter"]}
           , "formatters":
                {"simpleFormatter":
                    {"format": "%(asctime)s - %(name)s "
                               "- %(levelname)s - %(message)s"
                     }
                 }
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
            dataStoreExecuter: an Executor object that
                               holds threads used to
                               run FaceDataStore.genData()
    """
    clients: List[str] = []
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
            return Status(success=False, exitCode=e.exitCode)

        self.dataStore = FaceDataStore(cap, calibrated)
        self.dataStoreExecuter.submit(self.dataStore.genData)
        self.initialized = True

        logger_servicer.debug("Calibrated.")
        logger_servicer.debug(f"cap: {cap}")
        return Status(success=True, token=Token(token=secrets.token_hex()))

    def startStream(self, req, context):
        """Streams face data to the client
        """
        logger_servicer.info("called: startStream")
        if not self.dataStore.cap.isOpened():
            logger_servicer.info("camera isn't available")
            yield None

        self.clients.append(req.token)
        while req.token in self.clients:
            yield self.dataStore.current
        logger_servicer.debug("finished: startStream")

    def stopStream(self, req, context):
        """ stop streaming FaceData """
        logger_servicer.info("stopStream")

        if req.token in self.clients:
            self.clients.remove(req.token)

        logger_servicer.debug("Stream closed")
        return Status(success=True)

    def shutdown(self, req, context):
        """ shutdown this server
        """
        if 0 < len(self.clients):
            return Status(success=False, exitCode=ExitCode.FILE_MAIN
                                        | ExitCode.ServerIsStillUsed)

        self.dataStore.cap.release()
        return Status(success=True)
# }}}


def main():
    # Server setting
    server_address = "0.0.0.0"
    server_port = 5032
    multicast_group = '226.0.0.1'

    # Preparing camera
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()

    # ========== calibration ==========
    try:
        calib: RawFaceData = faceCalibration(cap)
    except FaceDetectionError as e:
        cap.release()
        logger_servicer.info(f"ERROR: Unexpected things are happened: {e}")
        logger_servicer.info("Aborting")
        return

    logger_servicer.debug("Calibrated.")
    logger_servicer.debug(f"cap: {cap}")

    try:
        # Preparing socket
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
            # Some options. See `man setsockopt'
            sock.setsockopt(sock.SOL_SOCKET
                           , sock.SO_BROADCAST, 1)  # enable BROADCAST
            sock.setsockopt(sock.IPPROTO_IP, sock.IP_MULTCAST_IF
                           , socket.inet_aton(server_address))

            # ========== Main loop ==========
            while True:
                if cap.isOpened() is not True:
                    break

                face: Face = Face.default()
                rots: FaceRotations = FaceRotations(0, 0, 0)
                _, frame = cap.read()
                landmark: Optional[dlib.points] = facemark(frame)

                face: Face          = Face.default()\
                                        if landmark is None\
                                        else Face.fromDPoints(landmark)

                rots: FaceRotations = FaceRotations(0, 0, 0)\
                                        if landmark is None\
                                        else FaceRotations.get(face, calib)

                sock.send(toBinary(generatedData))



    except KeyboardInterrupt:
        cap.release()


if __name__ == '__main__':
    main()
