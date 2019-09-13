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
            rl = "right" if 0 < fd.Y else "left"
            ud = "upside" if 0 < fd.X else "downside"
            print(f"Face faces {ud} {rl}")
    except KeyboardInterrupt:
        print("--- Calling stopStream")
        stub.stopStream(VoidCom())

    print("Done")
    sys.exit(0)


if __name__ == '__main__':
    main()