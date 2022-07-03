import numpy as np


TYPE_MAPE = {
    "FP32": np.float32,
    "UINT8": np.uint8
}


class NodeInfo():
    def __init__(self, node_name='input', node_type='FLOAT32', node_shape=[], node_data=None):
        self.node_name = node_name
        self.node_type = node_type
        self.node_shape = node_shape
        self.node_data = node_data

    def from_dict(self, node_dict: dict):
        # must have node_name
        self.node_name = node_dict['node_name']
        if 'node_type' in node_dict:
            self.node_type = node_dict['node_type']
        if 'node_shape' in node_dict:
            self.node_shape = node_dict['node_shape']
        if 'node_data' in node_dict:
            # data = self.convert_data_type(node_dict['node_data'], node_dict['node_type'])
            self.node_data = node_dict['node_data']

    def convert_data_type(self, data, type_name):
        new_data = data.astype(TYPE_MAPE[type_name])
        return new_data

    def reshape_data(self, data, shape):
        new_data = data.reshape(shape)
        return new_data
        