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


class SoftQNetwork(nn.Module):
    def __init__(self):
        super(SoftQNetwork, self).__init__()
        self.alpha = 4
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getV(self, q_value):
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value/self.alpha), dim=1, keepdim=True))
        return v
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print('state : ', state)
        with torch.no_grad():
            q = self.forward(state)
            v = self.getV(q).squeeze()
            # print('q & v', q, v)
            dist = torch.exp((q-v)/self.alpha)
            # print(dist)
            dist = dist / torch.sum(dist)
            # print(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    onlineQNetwork = SoftQNetwork().to(device)
    targetQNetwork = SoftQNetwork().to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    GAMMA = 0.99
    REPLAY_MEMORY = 50000
    BATCH = 16
    UPDATE_STEPS = 4

    expert_memory_replay = Memory(REPLAY_MEMORY//2)
    expert_memory_replay.load('expert_replay')
    online_memory_replay = Memory(REPLAY_MEMORY//2)
    writer = SummaryWriter('logs/sqil')

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        for time_steps in range(200):
            action = onlineQNetwork.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            online_memory_replay.add((state, next_state, action, 0., done))

            if online_memory_replay.size() > 1280:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

                online_batch = online_memory_replay.sample(BATCH//2, False)
                online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = zip(*online_batch)

                online_batch_state = torch.FloatTensor(online_batch_state).to(device)
                online_batch_next_state = torch.FloatTensor(online_batch_next_state).to(device)
                online_batch_action = torch.FloatTensor(online_batch_action).unsqueeze(1).to(device)
                online_batch_reward = torch.FloatTensor(online_batch_reward).unsqueeze(1).to(device)
                online_batch_done = torch.FloatTensor(online_batch_done).unsqueeze(1).to(device)

                expert_batch = expert_memory_replay.sample(BATCH//2, False)
                expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = zip(*expert_batch)

                expert_batch_state = torch.FloatTensor(expert_batch_state).to(device)
                expert_batch_next_state = torch.FloatTensor(expert_batch_next_state).to(device)
                expert_batch_action = torch.FloatTensor(expert_batch_action).unsqueeze(1).to(device)
                expert_batch_reward = torch.FloatTensor(expert_batch_reward).unsqueeze(1).to(device)
                expert_batch_done = torch.FloatTensor(expert_batch_done).unsqueeze(1).to(device)

                batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
                batch_next_state = torch.cat([online_batch_next_state, expert_batch_next_state], dim=0)
                batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
                batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
                batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)


                with torch.no_grad():
                    next_q = targetQNetwork(batch_next_state)
                    next_v = targetQNetwork.getV(next_q)
                    y = batch_reward + (1 - batch_done) * GAMMA * next_v

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), global_step=learn_steps)
            
            if done:
                break
            
            state = next_state
        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        if epoch % 10 == 0:
            torch.save(onlineQNetwork.state_dict(), 'sqil-policy.para')
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))




