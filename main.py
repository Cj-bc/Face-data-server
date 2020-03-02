import cv2
import dlib
from contextlib import closing
import secrets
import socket
from concurrent import futures
from typing import (Optional, List)

from FaceDataServer.faceDetection import (faceCalibration, facemark)
from FaceDataServer.Types import (RawFaceData,
                                 FaceDetectionError, Face, ExitCode,
                                 FaceData)
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
            sock.setsockopt(socket.SOL_SOCKET
                           , socket.SO_BROADCAST, 1)  # enable BROADCAST
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF
                           , socket.inet_aton(server_address))

            # ========== Main loop ==========
            while True:
                if cap.isOpened() is not True:
                    break

                _, frame = cap.read()
                landmark: Optional[dlib.points] = facemark(frame)

                face: Face          = Face.default()\
                                        if landmark is None\
                                        else Face.fromDPoints(landmark)

                data: FaceData = FaceData.default()\
                                        if landmark is None\
                                        else FaceData.get(face, calib)

                sock.sendto(data.toBinary(), (multicast_group, server_port))

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == '__main__':
    main()
