#!/usr/bin/env python

# This file contains the DQN algorithm implementations
# for the NEL reinforcement learning project.

from __future__ import print_function, division
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple, deque
import gym
import sys
import os
import time
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from utils.network_utils import *
from utils.algo_utils import *
from networks import *

class DQNAgent():

    def __init__(self, dqn, memory_size=50000, 
                 burn_in=10000, epsilon_start=1, 
                 epsilon_end=0.01, epsilon_strategy='decay',
                 epsilon=0.05, window=20, dqn_algo='vanilla',
                 conv_input=False):
        self.env = dqn.env
        self.env_name = self.env.spec.id
        self.dqn = dqn
        self.dqn_algo = dqn_algo
        self.target_dqn = deepcopy(self.dqn)
        self.device = self.dqn.device
        # True if convolutional layers are used
        self.conv_input = conv_input

        print("Network")
        print(self.dqn)
        print("Target Network")
        print(self.target_dqn)
        
        self.window = window # Moving average window to determine convergence
        self.replayMemory = replayMemory(memory_size=memory_size, burn_in=burn_in)
        self.reward_limit = self.env.spec.reward_threshold
        
        self.epsilon_strategy = epsilon_strategy
        if epsilon_strategy == 'decay':
            self.epsilon = copy(epsilon_start)
            self.epsilon_start = epsilon_start
            self.epsilon_end = epsilon_end
        elif epsilon_strategy == 'constant':
            self.epsilon = epsilon
        self.actions = np.arange(self.dqn.n_outputs)

        # Add attributes to save model
        self.env_name = self.env.spec.id
        self.algo_class = 'dqn'
        self.algo = self.dqn_algo 
        self.timestamp = time.strftime('%Y%m%d_%H%M')

    def greedyPolicy(self, state):
        qval, action = self.dqn.getAction(state)
        return action.item()

    def randomPolicy(self):
        return np.random.choice(self.actions)

    def getAction(self, state, mode='train'):
        # Training mode executes exploration
        if mode == 'train':
            if np.random.random() < self.epsilon:
                action = self.randomPolicy()
            else:
                action = self.greedyPolicy(state)
        # Randomly explore the state space
        elif mode == 'explore':
            action = self.randomPolicy()
        elif mode == 'test':
            action = self.greedyPolicy(state)

        return action

    def takeStep(self, mode='train'):
        action = self.getAction(self.s_0, mode=mode)
        s_1, r, done, _ = self.env.step(action)
        self.replayMemory.append(self.s_0, action, r, done, s_1.copy())
        self.s_0 = s_1.copy()

        if mode == 'train': 
            self.reward += r
            self.step_count += 1
        return done

    def train(self, gamma=0.99, max_episodes=3000, batch_size=32,
              update_freq=1, network_sync_freq=10, test_freq=200,
              print_episodes=False):
        self.gamma = gamma
        self.reward = 0
        self.batch_size = batch_size
        self.training_rewards, self.std_rewards = [], []
        self.training_step_rewards, self.training_loss = [], []
        self.ep_loss = []
        self.test_rewards = []
        self.test_ep_list = []
        self.test_mean_rewards = []
        # Count steps to update networks
        self.update_count, self.step_count, ep = 0, 0, 0

        # Train network
        training = True
        self.success = False
        self.s_0 = self.env.reset()

        # Populate replay buffer
        while self.replayMemory.burn_in_capacity() < 1:
            done = self.takeStep(mode='explore')
            if done:
                self.s_0 = self.env.reset()

        print("Training:")
        while training:
            self.reward = 0 
            done = False
            self.s_0 = self.env.reset()
            while done == False:
                # Decay epsilon
                if self.epsilon_strategy == 'decay':
                    self.epsilon = max(self.epsilon_start / 
                        ((ep + 1 ) / (max_episodes + 1)),
                                       self.epsilon_end)
                # Take Step
                done = self.takeStep(mode='train')

                # Update Networks
                if self.step_count % update_freq == 0:
                    self.update()
                # Sync Networks
                if self.step_count % network_sync_freq == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())

                # End of episode
                if done:
                    ep += 1
                    # TODO: Determine and implement testing strategy
                    self.training_rewards.append(self.reward)
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    std_rewards = np.std(self.training_rewards[-self.window:])
                    self.std_rewards.append(std_rewards)
                    self.training_loss.append(np.mean(self.ep_loss))
                    self.ep_loss = []

                    if print_episodes:
                        print("\rEpisode {:d} Mean Rewards: {:.2f}\t".format(
                            ep, mean_rewards), end="")

                        if mean_rewards >= self.reward_limit and self.success == False:
                                print("\nTarget reached")
                        elif mean_rewards <= self.reward_limit and self.success == True:
                                print("\nPerformance dropped below target.")

                    if mean_rewards >= self.reward_limit:
                        self.success = True
                    else:
                        self.success = False
                    
                    if ep >= max_episodes:
                        training = False
                        #self.test(test_episodes=100)
                        break

    def test(self, test_episodes=100):
        for ep in range(test_episodes):
            done = False
            self.reward = 0
            self.s_0 = self.env.reset()
            while done == False:
                done = self.takeStep(mode='test')
                if done:
                    self.test_rewards.append(self.reward)

    def calcLoss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        rewards_t = torch.FloatTensor(rewards).to(device=self.device)
        dones_t = torch.ByteTensor(dones).to(device=self.device)

        qvals = torch.gather(
            self.dqn.getQValues(states).to(device=self.device), 1, actions_t).squeeze()
        
        if self.dqn_algo == 'double' or self.dqn_algo =='dueling':
            _qvals, max_actions = self.target_dqn.getAction(next_states)
            next_qvals = self.target_dqn.getQValues(next_states)[range(
                len(max_actions)), max_actions].detach()
        else:
            next_qvals = torch.max(
                self.target_dqn.getQValues(next_states).to(device=self.device),
                dim=-1)[0].detach()
        # Set next states where the episode has terminated to value of 0
        next_qvals[dones_t] = 0
        expected_qvals = self.gamma * next_qvals + rewards_t

        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss
    
    def update(self):
        self.dqn.optimizer.zero_grad()
        batch = self.replayMemory.sample_batch(batch_size=self.batch_size)
        loss = self.calcLoss(batch)
        loss.backward()
        self.dqn.optimizer.step()
        self.ep_loss.append(torchToNumpy(loss, device=self.device))


