
import numpy as np 
from collections import OrderedDict, namedtuple, deque
from os import path, makedirs
import torch
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

def flattenDict(odict):
    # TODO: update to handle tau values greater than 1
    if type(odict) == OrderedDict or type(odict) == dict:
        return np.hstack([np.array(odict[i]).astype('float').flatten() 
            for i in odict.keys()])
    else:
        return odict

def processState(model, state):
    try:
        if model.conv_input:
            state_dict = {}
            zero_states = False
            for t in range(model.tau):
                if t >= len(state):
                    t = len(state) - 1
                    zero_states = True
                s = state[t]
                if type(s) is not dict and type(s) is not OrderedDict:
                    s = s[0]
                for key in s.keys():
                    if key == 'vision':
                        val = s[key].reshape(3, 11, -1)
                    else:
                        try:
                            val = s[key].astype(float)
                        except AttributeError:
                            val = float(s[key])
                    if zero_states:
                        if type(val) == float:
                            val = 0
                        else:
                            val = np.zeros(val.shape)
                    try:
                        state_dict[key].append(val)
                    except KeyError:
                        state_dict[key] = [val]
            
            for k in state_dict.keys():
                state_dict[k] = torch.FloatTensor(
                    np.stack(state_dict[k])).to(device=model.device)

            # Load tensors in dict
            state_t = {'vision': state_dict['vision']}
            state_t['other'] = torch.cat([state_dict['moved'].flatten(), 
                                          state_dict['scent'].flatten()])

            return state_t
    except AttributeError:
        pass
    
    return flattenDict(state[-1])
    
class batchContainer():

    def __init__(self):
        self.batch = namedtuple('batch', ['state', 
            'action', 'reward', 'next_state', 'mean_qval', 'done'])
        self.buffer = deque()

    def unpackBatch(self):
        return zip(*[self.buffer[i] for i in range(len(self.buffer))])

def discountRewards(rewards, gamma=0.99, baseline=False):
    '''
    Discounts rewards and returns discounted array. Options enable mean
    of discounted rewards to be used as a baseline.
    '''
    disc_rewards = np.array([gamma ** t * rewards[t]
        for t in range(len(rewards))])
    disc_rewards = disc_rewards[::-1].cumsum()[::-1] - 0
    if baseline:
        return disc_rewards - disc_rewards.mean()
    else:
        return disc_rewards

def discountBatchedRewards(rewards, dones, n=1, gamma=1):
    total_steps = len(rewards)
    ep_ends = np.where(dones==True)[0]
    completed_eps = 0
    G = np.zeros_like(rewards)
    for t in range(total_steps):
        if len(ep_ends) == 0:
            last_step = min(n, total_steps - t)
        else:
            last_step = min(n, ep_ends[completed_eps] - t)
            if t >= ep_ends[completed_eps]:
                completed_eps += 1
        G[t] = sum([rewards[t+i:t+i+1]*gamma**i for i in range(last_step)])

    return G

def saveModelWeights(agent):
    # Create directories depending on whether or not the agent reached its
    # success criteria, then move checkpoint files into that folder
    file = 'weights.pt'
    if agent.algo_class == 'dqn':
        net = deepcopy(agent.dqn)
    if agent.algo_class == 'policy_gradient':
        net = deepcopy(agent.actor)

    if agent.success:
        _filepath = path.join(agent.env_name, 'success', agent.algo_class, \
            agent.algo, agent.timestamp)
        makedirs(_filepath, exist_ok=True)
        filepath = path.join(_filepath, file)
        torch.save(net.state_dict(), filepath)
    else:
        _filepath = path.join(agent.env_name, 'failure', agent.algo_class, \
            agent.algo, agent.timestamp)
        makedirs(_filepath, exist_ok=True)
        filepath = path.join(_filepath, file)
        torch.save(net.state_dict(), filepath)
    
    # Save second network (if any)
    # TODO: update for A2C and other algorithms
    if agent.algo == 'dueling':
            v_net_path = path.join(_filepath, 'value_' + file)
            torch.save(target.state_dict(), v_net_path)
    # print("Checkpoint Files")
    # print(model.temp_files)
    # # Move temp files to appropriate folder
    # for file in model.temp_files:
    #     # if exit_status:
    #     new_file_path = path.join(model.filepath, path.basename(file))
    #     os.rename(file, new_file_path)
    # try:
    #     # Delete old files
    #     print("Deleting old files")
    #     shutil.rmtree(path.dirname(file))
    # except FileNotFoundError:
    #     pass

    print("\nModel saved to: {:s}".format(filepath))

    return _filepath

def saveResults(agent, args):
    filepath = saveModelWeights(agent)
    file = open(path.join(filepath, 'parameters.txt'), 'w')
    file.writelines('parameters,value')
    [file.writelines('\n' + str(k) + ',' + str(v)) 
        for k, v in vars(args).items()]
    file.close()
    # Save rewards
    vals_to_save = ['training_rewards', 'test_rewards', 
        'training_step_rewards', 'training_loss', 'kl_divergence', 
        'policy_loss', 'value_loss', 'entropy_loss']
    for val in vals_to_save:
        if hasattr(agent, val):
            data = getattr(agent, val)
            col_name = val.split('_')[-1]
            df = pd.DataFrame(data, columns=[col_name])
            df.to_csv(path.join(filepath, val + '.txt'))
            print('{:s} saved'.format(val))

    return filepath

class replayMemory():

    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.
        # Burn in episodes define the number of episodes that are written 
        # into the memory from the randomly initialized agent.
        # Memory size is the maximum size after which old elements in the 
        # memory are replaced. 
        self.memory_size = memory_size
        self.burn_in = burn_in
        assert self.memory_size >= self.burn_in
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'next_state', 'qvals', 'done'])
        self.replayMemory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions 
        # - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        # Get random indices from buffer
        samples = np.random.choice(len(self.replayMemory), batch_size, replace=False)
        # Use asterisk operator to unpack deque: 
        # https://medium.com/understand-the-python/understanding-the-asterisk-of-python-8b9daaa4a558
        batch = zip(*[self.replayMemory[i] for i in samples])
        return batch

    def append(self, state, action, reward, next_state, qvals, done):
        # Appends results to the memory buffer
        self.replayMemory.append(
            self.Buffer(state, action, reward, next_state, qvals, done))

    def burn_in_capacity(self):
        return len(self.replayMemory) / self.burn_in
    
    def capacity(self):
        # Returns the percentage of the buffer that is occupied
        return len(self.replayMemory) / self.memory_size

def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False 
    else:
        raise ArgumentTypeError('Boolean value expected.')
        
def plot_training_results(agent, filepath):
    window = agent.window
    vals_to_plot = ['training_step_rewards', 'test_rewards', 
        'training_loss', 'kl_divergence',
        'policy_loss', 'entropy_loss', 'value_loss']
    val_count = 0
    data_dict = {}
    for val in vals_to_plot:
        if hasattr(agent, val):
            data = getattr(agent, val)
            if len(data) == 0:
                continue
            mean_data = [np.mean(data[t - window:t]) 
                if t > window
                else np.mean(data[:t + 1]) 
                for t, d in enumerate(data)]
            title = val.replace('_', ' ').title()
            data_dict[val] = {'title': title,
                'data': data,
                'mean_data': mean_data}
            val_count += 1

    n_rows = int((val_count - val_count % 2) / 2 + 1)
    n_cols = 2
    row_lim = 0
    if val_count % 2 == 0:
        # If even, first two rows take both columns
        row_lim = 1
    width_ratios = [2 if w > row_lim else 1 for w in range(n_rows)]

    fig = plt.figure(figsize=(12,8))
    grid = plt.GridSpec(n_rows, n_cols) #, width_ratios=width_ratios)
    current_row, current_col = 0, 0
    for i, d in enumerate(data_dict.keys()):
        #print(current_row, current_col)
        if i <= row_lim:
            ax = fig.add_subplot(grid[current_row, :])
            ax.plot(data_dict[d]['data'], linewidth=0.2)
            ax.plot(data_dict[d]['mean_data'])
            ax.set_title(data_dict[d]['title'])
            current_row += 1
            if i == 0:
                ax.set_xlabel('Steps')
        else:
            ax = fig.add_subplot(grid[current_row, current_col % n_cols])
            ax.plot(data_dict[d]['data'], linewidth=0.2)
            ax.plot(data_dict[d]['mean_data'])
            ax.set_title(data_dict[d]['title'])
            current_row += (current_col) % 2
            current_col += 1

    plt.tight_layout()
    fig.savefig(path.join(filepath, 'training_plots.png'))

    return fig
    
def plot_testing_results(filepath, test_ep_list, test_mean_rewards):
    plt.figure(figsize=(12,8))
    plt.plot(test_ep_list, test_mean_rewards, label='Mean Rewards',color='c')
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.title('Test rewards using policies from training')
    plt.legend()
    plt.savefig(path.join(filepath, 'testing_plots.png'))
    plt.show()