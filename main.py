import cv2
import dlib
from contextlib import closing
import secrets
import socket
from concurrent import futures
from typing import (Optional, List, Tuple)

from FaceDataServer.faceDetection import (faceCalibration, facemark)
from FaceDataServer.Types import (RawFaceData,
                                 FaceDetectionError, Face, ExitCode,
                                 FaceData, Cv2Image, Coord,
                                 defaultPortNumber, defaultGroupAddr
                                  )
from logging import getLogger, Logger
import logging.config as loggingConfig
import numpy as np
import os
from Debug import face2Image

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
    server_port = defaultPortNumber
    multicast_group = defaultGroupAddr

    DEBUG = True if os.getenv('DEBUG', "NOTSET") != "NOTSET"\
                 else False

    # Preparing camera
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
    videoSize = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                , cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                 )

    # ========== calibration ==========
    try:
        calib, initialRatio = faceCalibration(cap)
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

                face, ratio = Face.defaultWithRatio(initialRatio)\
                               if landmark is None\
                               else Face.fromDPointsWithRatio(landmark)
                face.fixWithRatio(initialRatio, ratio)

                data: FaceData = FaceData.default()\
                                        if landmark is None\
                                        else FaceData.get(face, calib)

                sock.sendto(data.toBinary(), (multicast_group, server_port))

                if DEBUG:
                    cv2.imshow('face wire test'
                              , face2Image(videoSize, face, frame))
                    cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
