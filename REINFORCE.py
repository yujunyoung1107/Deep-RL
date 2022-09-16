import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical
from Network.MLP import MLP
from Memory.Memory import Buffer
from Utils.utils import *


class REINFORCE(nn.Module):

    def __init__(self, s_dim, a_dim, device):
        super(REINFORCE, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.policy = MLP(s_dim, a_dim, num_neurons=[256]).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.softmax = nn.Softmax(dim=-1)
        self.logmax = nn.LogSoftmax(dim=-1)
        self.memory = Buffer()
        self.gamma = 0.99
        self.device = device


    def get_action(self, state):

        state = ToTensor(state).to(self.device)
        prob = self.softmax(self.policy(state))
        action = Categorical(prob).sample()

        return action.view(-1).detach().cpu().numpy()

    def get_sample(self, s, a, r, ns, done):

        s = ToTensor(s)
        a = ToTensor(a, False)
        r = ToTensor(r)
        ns = ToTensor(ns)
        done = ToTensor(done)

        self.memory.push(s, a, r, ns, done)

    def update(self):
        
        states, actions, rewards, _, _ = self.memory.get_sample()

        G = 0.0
        for s, a, r in zip(states[::-1], actions[::-1], rewards[::-1]):
            s, a, r = s.to(self.device), a.to(self.device), r.to(self.device)
            G = r + self.gamma*G

            log_prob = self.logmax(self.policy(s)).view(-1)[a.item()]
            Loss = -log_prob*G

            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()

        self.memory.reset()

    def episodic_update(self):

        states, actions, rewards, _, _ = self.memory.get_sample()

        G = 0.0
        Reward2Go = []
        for r in rewards[::-1]:
            G = r + self.gamma*G
            Reward2Go.insert(0,G)

        Reward2Go = torch.tensor(Reward2Go).float().to(self.device)
        Reward2Go = (Reward2Go - Reward2Go.mean()) / Reward2Go.std()
        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.cat(actions, dim=0).to(self.device)

        log_prob = self.logmax(self.policy(states))
        log_prob = log_prob.gather(1, actions).view(-1)
        Loss = -(log_prob * Reward2Go).mean()

        self.optimizer.zero_grad()
        Loss.backward()
        self.optimizer.step()
        
        self.memory.reset()

    def run_episode(self, env, num_episode):

        reward_sum = []

        for ep in range(num_episode):
            cum_r = 0
            s = env.reset()[0]

            while True:
                a = self.get_action(s)
                ns, r, done, trunc, _ = env.step(a.item())
                new_done = False if (done==False and trunc==False) else True

                self.get_sample(s, a, r/100, ns, new_done)
                s = ns
                cum_r += r

                if new_done:
                    self.episodic_update()
                    reward_sum.append(cum_r)
                    break

            print('ep : {} | reward : {}'.format(ep, cum_r))

            if ep % 10 == 0 and ep != 0:
                print('ep : {} | reward_avg : {}'.format(ep, np.mean(reward_sum)))
                reward_sum = []



def main():


    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    agent = REINFORCE(s_dim, a_dim, device)

    agent.run_episode(env, 1000)
        

if __name__ == "__main__":
    main()