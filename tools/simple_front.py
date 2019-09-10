# Please execute this from tools/ dir
# As the PATH manipulation is not written properly,
# it won't work if executed from other path
import sys
sys.path.append("../")
import grpc
from FaceDataServer.faceDataServer_pb2_grpc import FaceDataServerStub
from FaceDataServer.faceDataServer_pb2 import VoidCom


def main():
    channel = grpc.insecure_channel('localhost:50052')
    stub = FaceDataServerStub(channel)
    print("--- Initializing... Please watch server's stdout")
    initStat = stub.init(VoidCom())
    if not initStat.success:
        print("Initialization failed.")
        sys.exit(initStat.exitCode)

    print("Initialized")

    try:
        print("--- Calling startStream")
        for fd in stub.startStream(VoidCom()):
            rl = "right" if 0 < fd.y else "left"
            ud = "upside" if 0 < fd.x else "downside"
            print(f"Face faces {ud} {rl}")
    except KeyboardInterrupt:
        print("--- Calling stopStream")
        stub.stopStream(VoidCom())

    print("Done")
    sys.exit(0)


if __name__ == '__main__':
    main()
