from sql import SoftQNetwork
from itertools import count
import torch
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    onlineQNetwork = SoftQNetwork().to(device)
    onlineQNetwork.load_state_dict(torch.load('sql-policy.para'))
    episode_reward = 0

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        for time_steps in range(200):
            env.render()
            action = onlineQNetwork.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
            state = next_state
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
