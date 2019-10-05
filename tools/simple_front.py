# Please execute this from tools/ dir
# As the PATH manipulation is not written properly,
# it won't work if executed from other path
import sys
sys.path.append("../")  # noqa: E402
import grpc
import math
import curses
from FaceDataServer.faceDataServer_pb2_grpc import FaceDataServerStub
from FaceDataServer.faceDataServer_pb2 import VoidCom

def coordPad(x: float, y: float, p=None):
    """ make pad with coordinate axis  """
    padX = 20
    padY = 20
    if p is None:
        p = curses.newpad(padY, padX)
    else:
        p.erase()

    p.border()
    p.hline(int(padY / 2), 1, ord('-'), padX)
    p.vline(1, int(padX / 2), ord('|'), padY)

    p.addch(int((padY / 2) - x), int((padX /2) + y), '*')
    return p


def main(stdscr):
    channel = grpc.insecure_channel('localhost:50052')
    stub = FaceDataServerStub(channel)
    stdscr.addstr(0, 0, "--- Initializing... Please watch server's stdout")
    initStat = stub.init(VoidCom())
    if not initStat.success:
        stdscr.addstr(0, 0, "Initialization failed.")
        sys.exit(initStat.exitCode)

    print("Initialized")

    try:
        print("--- Calling startStream")
        p = coordPad(0, 0)
        for fd in stub.startStream(initStat.token):
            p = coordPad(fd.x, fd.y, p)
            stdscr.addstr(0, 0, f"x: {math.degrees(fd.x)}, "
                                f"y: {math.degrees(fd.y)}, "
                                f"z: {math.degrees(fd.z)}" )
            p.refresh(0, 0, 5, 5, 70, 100)
            stdscr.refresh()
    except KeyboardInterrupt:
        print("--- Calling stopStream")
        _ = stub.stopStream(initStat.token)

    stub.shutdown(VoidCom())
    print("Done")
    sys.exit(0)


if __name__ == '__main__':
    curses.wrapper(main)
