# File of utility functions for running various networks.

from collections import OrderedDict
from torch import nn 

def torchToNumpy(tensor, device):
    '''
    Converts tensors to numpy based on CPU or GPU use
    '''
    if device=='cuda':
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def getLayersAndNodes(model, *args, **kwargs):
    '''
    Checks network layer values to return a list of hidden nodes to fit to 
    given parameters
    '''
    if len(args) > 1:
        layer_defs = args[0]
        assert layer_defs[0] == model.n_inputs, \
            "Network input dimension does not match environment state."
        assert layer_defs[-1] == model.n_outputs, \
            "Network output dimension does not match environment action space."
    else:
        layer_defs = [model.n_inputs]
        [layer_defs.append(model.n_hidden_nodes) for i in range(1,model.n_hidden_layers)]
        layer_defs.append(model.n_outputs)
    return layer_defs

def buildNetwork(model, layer_defs, *args, **kwargs):
    '''
    Builds an OrderedDict of PyTorch layers allowing you to define a custom
    neural network or use the defaults in the model.
    '''
    layers = OrderedDict()
    
    for l, n in enumerate(layer_defs):
        # Define input layer
        if l == 0:
            layers[str(2*l)] = nn.Linear(
                layer_defs[l],
                layer_defs[l+1],
                bias=model.bias)
        # Define intermediate layers
        elif l < model.n_hidden_layers - 1:
            layers[str(2*l)] = nn.Linear(
                layer_defs[l],
                layer_defs[l+1],
                bias=model.bias)
        # Define output layer
        elif l == model.n_hidden_layers - 1:
            layers[str(2*l)] = nn.Linear(
                layer_defs[l],
                layer_defs[l+1],
                bias=model.bias)
        if l < model.n_hidden_layers - 1:
            layers[str(2*l+1)] = getActivationFunction(model)

    return layers

def getActivationFunction(model, *args, **kwargs):
    '''
    Returns activation function for a neural network.
    '''
    if model.activation_function == 'relu':
        act_func = nn.ReLU()
    elif model.activation_function == 'tanh':
        act_func = nn.Tanh()
    elif model.activation_function == 'elu':
        act_func = nn.ELU()
    elif model.activation_function == 'sigmoid':
        act_func = nn.Sigmoid()
    elif model.activation_function == 'selu':
        act_func = nn.SELU()
    else:
        print("No activation function match. Default to ReLU.")
        act_func = nn.ReLU()
    return act_func

def xavierInit(network):
    if type(network) == nn.Linear:
        nn.init.xavier_uniform_(network.weight)
        # if self.bias:
        #     network.bias.data.normal(0, 1)
        #     network.bias.data.fill_(0)