import sys
import tritonclient.grpc as grpcclient


class TritonServingClient():
    def __init__(self, serving_url):
        host, port = serving_url.split(':')
        self.host = host
        self.port = port
        self.client = self.set_client()

    def set_client(self):
        try:
            url = self.host + ':' + str(self.port)
            client = grpcclient.InferenceServerClient(url=url,
                                                      verbose=False,
                                                      ssl=False,
                                                      root_certificates=None,
                                                      private_key=None,
                                                      certificate_chain=None)
        except Exception as e:
            print("channel create failed: " + str(e))
            sys.exit()
        return client

    def set_inputs(self, inputs_node: list):
        """
        Args:
            inputs: a list of NodeInfo
        Returns:
            a list of input tensors
        """
        inputs_tensor = []
        for node in inputs_node:
            input_tensor = grpcclient.InferInput(node.node_name, node.node_data.shape, node.node_type)
            input_tensor.set_data_from_numpy(node.node_data)
            inputs_tensor.append(input_tensor)
        return inputs_tensor

    def set_outputs(self, outputs_node):
        outputs_tensor = []
        for node in outputs_node:
            output_tensor = grpcclient.InferRequestedOutput(node.node_name)
            outputs_tensor.append(output_tensor)
        return outputs_tensor

    def convert_outputs(self, results, outputs_node: list):
        """
        """
        outputs = {}
        for output_node in outputs_node:
            temp_output = results.as_numpy(output_node.node_name)
            outputs[output_node.node_name] = temp_output
        return outputs

    def inference(self, name: str, configs: dict) -> dict:
        """model inference

        Args:
            name: model name
            configs: a dict including inputs and outputs
        Returns:
            a output dict
        """
        assert 'inputs' in configs, 'The key "inputs" mast in configs'
        inputs = self.set_inputs(configs['inputs'])
        outputs = self.set_outputs(configs['outputs'])

        results = self.client.infer(model_name=name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=None,
                                    headers={'test': '1'})
        statistics = self.client.get_inference_statistics(model_name=name)
        if len(statistics.model_stats) != 1:
            print('FAILED: Inference Statistics')
            sys.exit(1)

        if 'outputs' in configs:
            outputs = self.convert_outputs(results, configs['outputs'])
        else:
            outputs = results
        return outputs
