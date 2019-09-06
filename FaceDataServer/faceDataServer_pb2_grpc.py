# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import FaceDataServer.faceDataServer_pb2 as protos_dot_faceDataServer__pb2


class FaceDataServerStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.init = channel.unary_unary(
        '/FaceDataServer.FaceDataServer/init',
        request_serializer=protos_dot_faceDataServer__pb2.VoidCom.SerializeToString,
        response_deserializer=protos_dot_faceDataServer__pb2.Status.FromString,
        )
    self.startStream = channel.unary_stream(
        '/FaceDataServer.FaceDataServer/startStream',
        request_serializer=protos_dot_faceDataServer__pb2.VoidCom.SerializeToString,
        response_deserializer=protos_dot_faceDataServer__pb2.FaceData.FromString,
        )
    self.stopStream = channel.unary_unary(
        '/FaceDataServer.FaceDataServer/stopStream',
        request_serializer=protos_dot_faceDataServer__pb2.VoidCom.SerializeToString,
        response_deserializer=protos_dot_faceDataServer__pb2.Status.FromString,
        )


class FaceDataServerServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def init(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def startStream(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def stopStream(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FaceDataServerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'init': grpc.unary_unary_rpc_method_handler(
          servicer.init,
          request_deserializer=protos_dot_faceDataServer__pb2.VoidCom.FromString,
          response_serializer=protos_dot_faceDataServer__pb2.Status.SerializeToString,
      ),
      'startStream': grpc.unary_stream_rpc_method_handler(
          servicer.startStream,
          request_deserializer=protos_dot_faceDataServer__pb2.VoidCom.FromString,
          response_serializer=protos_dot_faceDataServer__pb2.FaceData.SerializeToString,
      ),
      'stopStream': grpc.unary_unary_rpc_method_handler(
          servicer.stopStream,
          request_deserializer=protos_dot_faceDataServer__pb2.VoidCom.FromString,
          response_serializer=protos_dot_faceDataServer__pb2.Status.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'FaceDataServer.FaceDataServer', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
