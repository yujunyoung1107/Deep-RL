import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical
from Network.MLP import MLP
from Memory.Buffer import Buffer
from Utils.utils import *


class A2C(nn.Module):
    
    def __init__(self, s_dim, a_dim):
        super(A2C, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor = MLP(s_dim, a_dim, num_neurons=[256])
        self.critic = MLP(s_dim, 1, num_neurons=[256])
        self.softmax = nn.Softmax(dim=-1)
        self.logmax = nn.LogSoftmax(dim=-1)
        self.parameters = list(self.actor.parameters()) + list(self.critic.parameters())

        self.optimizer = torch.optim.Adam(self.parameters, lr=3e-4)
        self.memory = Buffer()
        self.gamma = 0.99


    def get_action(self, state):

        state = torch.tensor(state).view(-1, self.s_dim).float()
        prob = self.softmax(self.actor(state))
        action = Categorical(prob).sample()

        return action.view(-1).detach().numpy()

    def get_sample(self, s, a, r, ns, done):

        s = ToTensor(s)
        a = ToTensor(a, False)
        r = ToTensor(r)
        ns = ToTensor(ns)
        done = ToTensor(done)

        self.memory.push(s, a, r, ns, done)

    def update(self):

        s, a, r, ns, done = self.memory.get_sample()

        G = 0.0
        Reward2Go = []
        for R in r[::-1]:
            G = R.item() + self.gamma*G
            Reward2Go.insert(0, G)
        Reward2Go = torch.tensor(Reward2Go).view(-1,1).float()

        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)
        r = torch.cat(r, dim=0)
        ns = torch.cat(ns, dim=0)
        done = torch.cat(done, dim=0)

        value = self.critic(s)
        output = self.actor(s)
        prob = self.softmax(output)
        log_prob = self.logmax(output)
        entropy = -(prob*log_prob).sum(dim=-1, keepdim=True)

        Advantage = (Reward2Go - value).detach()
        log_prob = log_prob.gather(1,a)

        critic_loss = (value - Reward2Go).pow(2)
        actor_loss = -(log_prob*Advantage)
        total_loss = (critic_loss + actor_loss - 0.01*entropy).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.memory.reset()

def main():


    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = A2C(s_dim, a_dim)

    for ep in range(2000):
        cum_r = 0
        s = env.reset()
        while True:        
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())
            agent.get_sample(s, a, r/100, ns, done)
            s = ns
            cum_r += r

            if done:
                agent.update()
                break

        print('episode : {} | reward : {}'.format(ep, cum_r))
        

if __name__ == "__main__":
    main()


   
        
