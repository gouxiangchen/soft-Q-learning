from sql import SoftQNetwork
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from collections import deque
import random
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
import gym
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path):
        b = np.load(path+'.npy', allow_pickle=True)
        assert(b.shape[0] == self.memory_size)

        for i in range(b.shape[0]):
            self.add(b[i])


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    onlineQNetwork = SoftQNetwork().to(device)
    onlineQNetwork.load_state_dict(torch.load('sql-policy.para'))
    episode_reward = 0

    REPLAY_MEMORY = 25000
    memory_replay = Memory(REPLAY_MEMORY)
    # memory_replay.load('expert_replay')

    # batch = memory_replay.sample(16, False)
    # batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

    # batch_state = torch.FloatTensor(batch_state).to(device)
    # batch_next_state = torch.FloatTensor(batch_next_state).to(device)
    # batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
    # batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
    # batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)


    # print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_done.shape)

    # exit()

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        for time_steps in range(200):
            # env.render()
            action = onlineQNetwork.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            memory_replay.add((state, next_state, action, 1., done))
            episode_reward += reward
            if memory_replay.size() == REPLAY_MEMORY:
                print('expert replay saved...')
                memory_replay.save('expert_replay')
                exit()
            if done:
                break
            state = next_state
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
