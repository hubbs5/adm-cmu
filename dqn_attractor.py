import numpy as np
import matplotlib.pyplot as plt
import gym 
import torch
from torch import nn
import seaborn as sns
import pandas as pd
import os
import copy
import sys
from collections import namedtuple, deque, Iterable
import time

from utils.network_utils import *
from utils.algo_utils import *

class DQNAttractorAgent(object):
    
    def __init__(self, dqn, I_0=2.5, K=5, B=5, noise=0.01, max_t=1, memory_size=10000, burn_in=32):
        
        self.dqn = dqn
        self.env = self.dqn.env
        self.env_name = self.dqn.env.spec.id
        self.n_inputs = self.dqn.n_inputs
        self.n_actions = self.dqn.actions.shape[0]
        self.device = self.dqn.device
        
        self.I_0 = I_0 # Background excitation
        self.K = K # Cross Inhibition
        self.B = B # Cross Inhibition (BETA)
        self.noise = noise # Noise scale (GAMMA)
        self.max_t = max_t # Max time to decision (seconds)
        self.dt = 0.001 # Decision time step (seconds)
        self.time_steps = int(self.max_t / self.dt) # Integer number of steps till decision
        
        self.replayMemory = replayMemory(memory_size=memory_size, burn_in=burn_in)
        self.timestamp = time.strftime("%Y%m%d_%H%M")
        self.algo_class = 'dqn_attractor'
        
    def attractorDecision(self, state):
        # Firing rates
        r = torch.zeros((self.time_steps, self.n_actions))
        # Noise vectors for state
        E = self.noise * np.sqrt(self.dt) * np.random.randn(self.time_steps, self.n_inputs)
        # Set initial firing rate to I_0
        r[0] = self.I_0 
        mean_qval_est = torch.zeros(self.n_actions)
        for t in range(1, self.time_steps):
            # Get Q-Values
            #qvals = torchToNumpy(self.dqn.getQValues(state + E[t]), device=self.device)
            qvals = self.dqn.getQValues(state + E[t])
            mean_qval_est += (qvals - mean_qval_est) / (t + 1)
            # Update firing rates
            for n in range(self.n_actions):
                r[t, n] = r[t-1, n] + self.dt * (qvals[n] + self.I_0 + self.K*r[t-1, n] - self.B * \
                        sum([r[t-1, m] for m in range(self.n_actions) if n!=m]))
                
        return r, mean_qval_est
        
    def getAction(self, state):
        r, mean_qval_est = self.attractorDecision(state)
        return torch.argmax(r[-1]).item(), mean_qval_est
    
    def takeStep(self):
        action, qval = self.getAction(self.s_0)
        s_1, reward, done, _ = self.env.step(action)
        self.replayMemory.append(self.s_0, action, reward, s_1.copy(), done)
        self.s_0 = s_1.copy()
        self.ep_reward += reward
        
        return done
            
    def train(self, gamma=1, max_episodes=10000, batch_size=32, 
    	update_freq=1, network_sync_freq=10, print_episodes=False,
    	window=20):
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.window = window
        self.training_rewards, self.std_rewards = [], []
        self.training_loss, self.ep_loss = [], []
        self.update_count, ep_count, self.ep_reward, self.step_count = 0, 0, 0, 0
        self.s_0 = self.env.reset()
        
        # Populate replay buffer
        while self.replayMemory.burn_in_capacity() < 1:
            done = self.takeStep()
            if done:
                self.s_0 = self.env.reset()
        
        # Begin training
        training = True        
        while training:
            done = self.takeStep()
            if self.step_count % update_freq == 0:
                self.update()
                
            if done:
                self.s_0 = self.env.reset()
                self.training_rewards.append(self.ep_reward)
                self.training_loss.append(np.mean(self.ep_loss))
                self.ep_loss = []
                self.ep_reward = 0
                ep_count += 1
                
                if print_episodes:
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    print("\rEpisode {:d} Mean Rewards: {:.2f}\t".format(
                        ep_count, mean_rewards), end="")

                if ep_count >= max_episodes:
                	training = False
                	break
                    
    def calcLoss(self, batch):
        states, actions, rewards, next_states, dones = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        rewards_t = torch.FloatTensor(rewards).to(device=self.device)
        dones_t = torch.ByteTensor(dones).to(device=self.device)
    
        qvals = torch.gather(self.dqn.getQValues(states), 1, actions_t).squeeze()
        next_qvals_t = torch.max(self.dqn.getQValues(next_states), dim=-1)[0].detach()
        next_qvals_t[dones_t] = 0
        expected_qvals = self.gamma * next_qvals_t + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss
    
    def update(self):
        self.dqn.optimizer.zero_grad()
        batch = self.replayMemory.sample_batch(batch_size=self.batch_size)
        loss = self.calcLoss(batch)
        loss.backward()
        self.dqn.optimizer.step()
        self.ep_loss.append(torchToNumpy(loss, device=self.device))