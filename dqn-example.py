'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import glob, re, os
import pandas as pd
from pathlib import Path

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super().__init__()
        ## TODO ##
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        ## TODO ##
        return self.layers(x)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        self._criterion = nn.MSELoss()
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##

        if random.random() < epsilon:
             return action_space.sample()
        else:
            self._behavior_net.eval()
            with torch.no_grad():
                # state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
                # action_values = self._behavior_net(state_tensor.unsqueeze(0))
                # return action_values.argmax().item()
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_values = self._behavior_net(state)
                self._behavior_net.train()
                return np.argmax(action_values.cpu().data.numpy())

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO ##
        q_values = self._behavior_net(state).gather(1, action.long())
        with torch.no_grad():
            q_next = self._target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + gamma * q_next * (1 - done)
        loss = self._criterion(q_values, q_target)

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        #self._target_net.load_state_dict(self._behavior_net.state_dict())
        for target_param, param in zip(self._target_net.parameters(), self._behavior_net.parameters()):
            target_param.data.copy_(param.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        max_total_reward = -1000
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                
                if(max_total_reward < total_reward):
                    max_total_reward = total_reward
                    agent.save(args.logdir+f"/dqn{episode}.pth",True)
                    
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()

        ## TODO ##
        for t in itertools.count(start=1):
            
            env.render()
            action = agent.select_action(state, epsilon, action_space)
            next_state, reward, done, _ = env.step(action)
 
            state = next_state
            total_reward += reward

            if done:
                rewards.append(total_reward)
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f'Episode: {n_episode}\tTotal reward: {total_reward}')
                #state = env.reset()
                break
            
    #average_reward = np.mean(rewards)
    #writer.add_scalar('Test/Average Reward', average_reward)
    print('Average Reward', np.mean(rewards))
    env.close()

class Recorder():
    def __init__(self, path='', sep='', recordname='train_details'):
        self.save_path = self.increment_path(path, sep)
        self.recorder =  open(self.save_path+'\\'+recordname+'.txt', 'a') 
        #self.train_detail = []
        self.recordname = recordname
    
    # save all the namespace parameters in the file which name "recordname".txt
    def save_record(self, key, value):
        self.recorder.write(key + ': ' + str(value))

    # produce the increment path from the given path
    # modify from YoloV7 source code
    def increment_path(self, path, sep=''):
        path = Path(path)  # os-agnostic
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        os.makedirs(f"{path}{sep}{n}")
        return f"{path}{sep}{n}" # n, f"{path}{sep}{n}"

    # save training detail as a csv file
    # def save_csv(self):
    #     train_detail = np.array(recorder.train_detail)
    #     train_detail = pd.DataFrame(train_detail, columns = ['Epoch','Train loss','Valid loss', 'LR'])
    #     train_detail.to_csv(self.save_path+'\\'+self.recordname+'.csv',index=False)

def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn/exp')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    if not args.test_only:
        # change logdir path
        global recorder
        recorder = Recorder(path=args.logdir)
        args.logdir = recorder.save_path
        # save all the args in txt file
        recorder.recorder.write(str(args))
        #print(args.logdir)
    else:
        args.logdir='log/dqn'

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.logdir+'/'+args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
