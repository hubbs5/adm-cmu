#!/usr/bin/env python

# Call this program to train and test a DQN algorithm

from __future__ import print_function, division
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple, deque
import gym
from gym import wrappers
import sys
import os
import time
import matplotlib as mpl
# Set display variables in case of headless linux server
if sys.platform == 'linux':
    if os.environ.get('DISPLAY', '') == '':
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
import shutil
import warnings

import nel

from utils.network_utils import *
from utils.algo_utils import *
from networks import *
from algos.dqn_algo import *


def parseArguments():
    parser = ArgumentParser(description='Deep Q Network Argument Parser')
    # Network parameters
    parser.add_argument('--hl', type=int, default=3,
        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--hn', type=int, default=16,
        help='An integer number that defines the number of hidden nodes.')
    parser.add_argument('--lr', type=float, default=0.001,
        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--bias', type=str2bool, default=True,
        help='Boolean to determine whether or not to use biases in network.')
    parser.add_argument('--gpu', type=str2bool, default=False,
        help='Boolean to enable GPU computation. Default set to False.')
    parser.add_argument('--actFunc', type=str, default='relu',
        help='String to define activation function.')
    parser.add_argument('--convNet', type=str2bool, default=False,
        help='Utilize a convolutional neural network.')

    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    
    # Training arguments
    parser.add_argument('--gamma', type=float, default=0.99,
        help='A value between 0 and 1 to discount future rewards.')
    parser.add_argument('--maxEps', type=int, default=10000,
        help='An integer number of episodes to train the agent on.')
    parser.add_argument('--netSyncFreq', type=int, default=2000,
        help='An integer number that defines when to update the target network.')
    parser.add_argument('--updateFreq', type=int, default=1,
        help='Integer value that determines how many steps or episodes' + 
        'must be completed before a backpropogation update is taken.')
    parser.add_argument('--testFreq', type=int, default=200,
        help='Integer value that determines after how many episodes' + 
        'testing is performed on the current policy')
    parser.add_argument('--batch', type=int, default=32,
        help='An integer number that defines the batch size.')
    parser.add_argument('--memorySize', type=int, default=50000,
        help='An integer number that defines the replay buffer size.')
    parser.add_argument('--burnIn', type=int, default=20000,
        help='Set the number of random burn-in transitions before training.')
    parser.add_argument('--print', type=str2bool, default=True,
        help='True to print running averages, else False by default.')
    parser.add_argument('--mpcSteps', type=int, default=10,
        help='Set the number of steps the MPC will use to look ahead.')
    
    # DQNAgent Arguments
    parser.add_argument('--epsStart', type=float, default=0.5,
        help='Float value for the start of the epsilon decay.')
    parser.add_argument('--epsEnd', type=float, default=0.05,
        help='Float value for the end of the epsilon decay.')
    parser.add_argument('--epsStrategy', type=str, default='constant',
        help="Enter 'constant' to set epsilon to a constant value or 'decay'" + 
        "to have the value decay over time. If 'decay', ensure proper" + 
        "start and end values.")
    parser.add_argument('--epsConstant', type=float, default=0.05,
        help='Float to be used in conjunction with a constant epsilon strategy.')
    parser.add_argument('--window', type=int, default=100,
        help='Integer value to set the moving average window.')
    parser.add_argument('--dqnAlgo', type=str, default='vanilla',
        help='Use vanilla DQN, double or dueling DQN.')
    parser.add_argument('--plot', type=str2bool, default=False,
        help='If true, plot training results.')
    parser.add_argument('--mpc', type=str2bool, default=False,
        help='Uses MPC if available.')
    parser.add_argument('--randomMPC', type=str2bool, default=False,
        help='Enables random sampling from the MPC trajectories.')
    args = parser.parse_args()

    return parser.parse_args()

def main(argv):

    args = parseArguments()
    #print(args)
    if args.gpu is None or args.gpu == False:
        args.gpu = 'cpu'
    else:
        args.gpu = 'cuda'

    env = gym.make(args.env)
    # Initialize DQNetwork
    if args.convNet:
        dqn = convQNetwork(env=env, 
            n_hidden_layers=args.hl, 
            n_hidden_nodes=args.hn, 
            learning_rate=args.lr, 
            bias=args.bias,
            device=args.gpu,
            dqn_algo=args.dqnAlgo,
            activation_function=args.actFunc,
            tau=args.tau)
    else:
        dqn = QNetwork(env=env, 
            n_hidden_layers=args.hl, 
            n_hidden_nodes=args.hn, 
            learning_rate=args.lr, 
            bias=args.bias,
            device=args.gpu,
            dqn_algo=args.dqnAlgo,
            activation_function=args.actFunc)
    # Initialize DQNAgent
    agent = DQNAgent(dqn,
        memory_size=args.memorySize,
        burn_in=args.burnIn, 
        epsilon_start=args.epsStart,
        epsilon_end=args.epsEnd,
        epsilon_strategy=args.epsStrategy,
        epsilon=args.epsConstant, 
        window=args.window,
        dqn_algo=args.dqnAlgo,
        conv_input=args.convNet)
    print("Running DQN for {:s}".format(args.env))
    # [print(str(k) + ' = ' + str(v)) for k, v in vars(args).items()]
    agent.train(gamma=args.gamma, 
        max_episodes=args.maxEps,
        batch_size=args.batch,
        update_freq = args.updateFreq,
        network_sync_freq=args.netSyncFreq,
        test_freq=args.testFreq,
        print_episodes=args.print)
    fp = saveResults(agent, args)
    fig = plot_training_results(agent, fp)
    if args.plot:
        plt.show()

    return agent.success

if __name__ == '__main__':
    start_time = time.time()
    success = main(sys.argv)
    end_time = time.time()
    x = end_time - start_time
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    # print("Agent training: {}".format(success))
    print("\nTraining Time: {:02}:{:02}:{:02}\n".format(int(hours), int(minutes), int(seconds)))
