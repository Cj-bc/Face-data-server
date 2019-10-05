# Please execute this from tools/ dir
# As the PATH manipulation is not written properly,
# it won't work if executed from other path
import sys
sys.path.append("../")  # noqa: E402
import grpc
import curses
from FaceDataServer.faceDataServer_pb2_grpc import FaceDataServerStub
from FaceDataServer.faceDataServer_pb2 import VoidCom


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
        for fd in stub.startStream(initStat.token):
            rl = "right" if 0 < fd.y else "left"
            ud = "upside" if 0 < fd.x else "downside"
            print(f"Face faces {ud} {rl}")
    except KeyboardInterrupt:
        print("--- Calling stopStream")
        _ = stub.stopStream(initStat.token)

    stub.shutdown(VoidCom())
    print("Done")
    sys.exit(0)


if __name__ == '__main__':
    curses.wrapper(main)
