#!/usr/bin/env python

import sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import gym 
import torch
from torch import nn
import seaborn as sns
import pandas as pd
import os
import copy
import time
from argparse import ArgumentParser, ArgumentTypeError

from dqn_attractor import *
from networks import *
from utils.network_utils import torchToNumpy

def parseArguments():
    parser = ArgumentParser(description='Deep Q Network Argument Parser')
    # Network parameters
    parser.add_argument('--hl', type=int, default=2,
        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--hn', type=int, default=8,
        help='An integer number that defines the number of hidden nodes.')
    parser.add_argument('--lr', type=float, default=0.01,
        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--bias', type=str2bool, default=False,
        help='Boolean to determine whether or not to use biases in network.')
    parser.add_argument('--gpu', type=str2bool, default=False,
        help='Boolean to enable GPU computation. Default set to False.')
    parser.add_argument('--actFunc', type=str, default='relu',
        help='String to define activation function.')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    
    # Training arguments
    parser.add_argument('--gamma', type=float, default=1,
        help='A value between 0 and 1 to discount future rewards.')
    parser.add_argument('--maxEps', type=int, default=5000,
        help='An integer number of episodes to train the agent on.')
    parser.add_argument('--netSyncFreq', type=int, default=100,
        help='An integer number that defines when to update the target network.')
    parser.add_argument('--updateFreq', type=int, default=1,
        help='Integer value that determines how many steps or episodes' + 
        'must be completed before a backpropogation update is taken.')
    parser.add_argument('--batch', type=int, default=32,
        help='An integer number that defines the batch size.')
    parser.add_argument('--memorySize', type=int, default=10000,
        help='An integer number that defines the replay buffer size.')
    parser.add_argument('--burnIn', type=int, default=32,
        help='Set the number of random burn-in transitions before training.')
    parser.add_argument('--print', type=str2bool, default=True,
        help='True to print running averages, else False by default.')
    parser.add_argument('--window', type=int, default=20,
        help='True to plot results.')
    parser.add_argument('--plot', type=str2bool, default=True,
        help='True to plot results.')
    
    # DQNAgent Arguments
    parser.add_argument('--I0', type=float, default=2.5,
    	help='Background excitation level.')
    parser.add_argument('--K', type=float, default=5.0,
    	help='Direct excitation gain.')
    parser.add_argument('--B', type=float, default=5.0,
    	help='Indirect excitation gain.')
    parser.add_argument('--noise', type=float, default=0.01,
    	help='Noise scalar.')
    parser.add_argument('--max_t', type=float, default=1.0,
    	help='Maximum time to decision')
    args = parser.parse_args()

    return parser.parse_args()

def main(argv):

    args = parseArguments()
    print(args)
    if args.gpu is None or args.gpu == False:
        args.gpu = 'cpu'
    else:
        args.gpu = 'cuda'

    env = gym.make(args.env)
    # Initialize DQNetwork
    dqn = QNetwork(env=env, 
        n_hidden_layers=args.hl, 
        n_hidden_nodes=args.hn, 
        learning_rate=args.lr, 
        bias=args.bias,
        device=args.gpu,
#        dqn_algo=args.dqnAlgo,
        activation_function=args.actFunc)
    # Initialize DQNAgent
    agent = DQNAttractorAgent(dqn,
    	I_0=args.I0,
    	K=args.K,
    	B=args.B,
    	noise=args.noise,
    	max_t=args.max_t,
        memory_size=args.memorySize,
        burn_in=args.burnIn)
    #print("Running DQN for {:s}".format(args.env))
    # [print(str(k) + ' = ' + str(v)) for k, v in vars(args).items()]
    agent.train(gamma=args.gamma, 
        max_episodes=args.maxEps,
        batch_size=args.batch,
        update_freq = args.updateFreq,
        network_sync_freq=args.netSyncFreq,
        print_episodes=args.print,
        window=args.window)
    fp = saveResults(agent, args)
    if args.plot:
    	plot_training_results(agent, fp)

if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    x = end_time - start_time
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    # print("Agent training: {}".format(success))
    print("\nTraining Time: {:02}:{:02}:{:02}\n".format(int(hours), int(minutes), int(seconds)))