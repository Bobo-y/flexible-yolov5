import numpy as np
import logging
import grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

try:
    from tensorflow.contrib.util import make_tensor_proto
except:
    # for tf2.x
    from tensorflow import make_tensor_proto

logger = logging.getLogger(__name__)

options = [('grpc.max_send_message_length', 256 * 1024 * 1024),
           ('grpc.max_receive_message_length', 256 * 1024 * 1024)]


class TFServingClient():
    """Tensorflow Serving Client

    Attributes:
      serving_url: like localhost:8500
    """

    def __init__(self, serving_url):
        """set host and port
        """
        host, port = serving_url.split(':')
        self.host = host
        self.port = int(port)
        self.channel = self.set_channel()

    def set_channel(self):
        """set a grpc channel using host and port

        Returns:
          a grpc channel
        """
        channel = grpc.insecure_channel(str(self.host) + ':' + str(self.port), options=options)
        return channel

    def set_stub(self):
        """set prediction_service_pb2_grpc stub

        Returns:
          a PredictionServiceStub
        """
        stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

        return stub

    def health_check(self, name, signature_name, version=None):
        """
        """
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = name
        request.model_spec.signature_name = signature_name
        if version:
            request.model_spec.version.value = version

        stub = model_service_pb2_grpc.ModelServiceStub(self.channel)
        try:
            response = stub.GetModelStatus(request, 10)
            if len(response.model_version_status) > 0:
                return True
        except Exception as err:
            logging.exception(err)
            return False

    def set_request(self, name, signature_name, feed_map, version=None):
        """format a PredictRequest

        Arguments:
          name: model_spec name
          signature_name: model_spec signature name
          feed_map: input data
          version: model_spec version, default is None
        Returns:
          a PredictRequest
        """
        request = predict_pb2.PredictRequest()
        request.model_spec.name = name
        request.model_spec.signature_name = signature_name
        # set version
        if version:
            request.model_spec.version.value = version

        for input_key, input_data in feed_map.items():
            request.inputs[input_key].CopyFrom(make_tensor_proto(input_data))
        return request

    def predict(self, request):
        """Generate a channel to predict request

        Arguments:
          request: a PredictRequest
        Returns:
          a PredictResponse
        """
        stub = self.set_stub()
        predict_result = stub.Predict.future(request, 10)
        temp_result = predict_result.result()
        return temp_result.outputs

    def convert_outputs(self, results, outputs_node: list):
        """
        """
        outputs = {}
        for node in outputs_node:
            temp_output = np.reshape(np.array(results[node.node_name].float_val), node.node_shape)
            outputs[node.node_name] = temp_output
        return outputs

    def check_configs(self, configs):
        requested_keys = ['inputs', 'model_spec_signature_name', 'outputs']
        for key in requested_keys:
            assert key in configs, f'Can not find {key} in configs.'

    def inference(self, name: str, configs: dict):
        try:
            outputs = self._inference(name, configs)
        except Exception as err:
            logger.error(err)
            raise err
        return outputs

    def _inference(self, name: str, configs: dict):
        """a simpoe wrapper for predict

        Args:
            name: model name
            configs: a dict including inputs and outputs
        Returns:
            a dict of outputs
        """
        self.check_configs(configs)

        feed_map = {}
        for node in configs['inputs']:
            feed_map[node.node_name] = node.node_data
        version = None if 'version' not in configs else configs['version']
        request = self.set_request(name, configs['model_spec_signature_name'], feed_map, version)
        results = self.predict(request)
        outputs = self.convert_outputs(results, configs['outputs'])
        return outputs
