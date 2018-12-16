import numpy as np
from random import seed

from .node import Node


class AritificalNeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
            self.num_inputs = num_inputs;
            self.network = self.__initialize_network__(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)

    def __compute_weighted_sum__(self, inputs, weights, bias):
        return np.sum(inputs * weights) + bias

    def __node_activation__(self, weighted_sum):
        return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

    def __initialize_network__(self, num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
        num_nodes_previous = num_inputs
        network = {}

        for layer in range(num_hidden_layers + 1):

            if layer == num_hidden_layers:
                layer_name = 'output'
                num_nodes = num_nodes_output
            else:
                layer_name = 'layer_{}'.format(layer + 1)
                num_nodes = num_nodes_hidden[layer]


            network[layer_name] = {}
            for node in range(num_nodes):
                node_name = 'node_{}'.format(node+1)
                network[layer_name][node_name] = Node(
                    np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                    np.around(np.random.uniform(size=1), decimals=2))

            num_nodes_previous = num_nodes

        return network

    def generate_random_inputs(self):
        np.random.seed(12)
        return np.around(np.random.uniform(size=self.num_inputs), decimals=2)

    def forward_propagate(self, inputs):
        layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer

        for layer in self.network:

            layer_data = self.network[layer]

            layer_outputs = []
            for layer_node in layer_data:

                node_data = layer_data[layer_node]

                # compute the weighted sum and the output of each node at the same time
                node_output = self.__node_activation__(self.__compute_weighted_sum__(layer_inputs, node_data.weights, node_data.bias))
                layer_outputs.append(np.around(node_output[0], decimals=4))

            if layer != 'output':
                print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

            layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

        network_predictions = layer_outputs
        return network_predictions
