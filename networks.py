import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple, deque
import gym
from copy import deepcopy
from utils.network_utils import *

class QNetwork(nn.Module):

    def __init__(self, env, n_hidden_layers=1, 
        n_hidden_nodes=4, learning_rate=0.01, bias=False, 
        activation_function='relu', dqn_algo='vanilla', 
        device='cpu', *args, **kwargs):
        super(QNetwork, self).__init__()

        algo_list = ['vanilla', 'double', 'dueling']
        self.dqn_algo = dqn_algo.lower()
        assert self.dqn_algo in algo_list, \
            "dqn_algo {} not recognized, provide one of: {}.".format(dqn_algo, algo_list)
            
        self.n_inputs = self.env.observation_space.shape[0]        
        self.n_outputs = self.env.action_space.n 
        # Allow custom layer definition
        if len(args) > 0:
            self.n_hidden_layers = len(args)
        else:
            self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.bias = bias
        self.actions = np.arange(self.n_outputs)
        self.learning_rate = learning_rate
        self.activation_function = activation_function.lower()
        self.device = device

        # Build network
        layer_list = getLayersAndNodes(self, args)
        self.layers = buildNetwork(self, layer_list)
        self.net = nn.Sequential(self.layers)
        self.net.apply(xavierInit)

        if dqn_algo == 'dueling':
            dueling_list = getLayersAndNodes(self, args)
            dueling_list[-1] = 1
            self.dueling_layers = buildNetwork(self, dueling_list)
            self.dueling_net = nn.Sequential(self.dueling_layers)
            self.dueling_net.apply(xavierInit)

        if self.device == 'cuda':
            self.net.cuda()
            if dqn_algo == 'dueling':
                self.dueling_net.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.learning_rate)
     
    def getQValues(self, state):
        state = flattenDict(state)
        try:
            state_t = torch.FloatTensor(state).to(device=self.device)
        except TypeError:
            print(len(state))
            print(state)
        if self.dqn_algo == 'vanilla':
            return self.net(state_t)
        elif self.dqn_algo == 'double':
            return self.net(state_t)
        elif self.dqn_algo == 'dueling':
            A = self.net(state_t)
            V = self.dueling_net(state_t)
            return V + A - A.mean()

    def getAction(self, state):
        qval, action = torch.max(self.getQValues(state), dim=-1)
        return qval.detach(), action.detach()