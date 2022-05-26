from stat import S_IMODE
import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from Network.MLP import MLP
from Memory.Memory import Buffer
from Utils.utils import *

class ActorCritic(nn.Module):

    def __init__(self, s_dim, a_dim, max_action):
        super(ActorCritic, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.max_action = max_action
        self.actor = MLP(s_dim, a_dim, num_neurons=[128], out_act = 'Tanh')
        self.critic = MLP(s_dim, 1, num_neurons=[128])
        self.log_std = nn.Parameter(0.0*torch.ones(1, a_dim))

    def policy(self, state):
        
        mu = self.actor(state) * self.max_action
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        return dist

    def value(self, state):

        return self.critic(state)

class PPO(nn.Module):
    
    def __init__(self, s_dim, a_dim, max_action):
        super(PPO, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.max_action = max_action
        self.network = ActorCritic(s_dim, a_dim, max_action)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=3e-4)
        self.memory = Buffer()
        self.gamma = 0.9
        self.rambda = 0.9
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.batch_size = 32
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):

        state = torch.tensor(state).view(1,-1).float()

        with torch.no_grad():
            dist = self.network.policy(state)
        
        action = dist.sample()

        return action.view(-1).numpy()

    def get_sample(self, s, a, r, ns, done):
        
        s = ToTensor(s)
        a = ToTensor(a, False)
        r = ToTensor(r)
        ns = ToTensor(ns)
        done = ToTensor(done)

        self.memory.push(s, a, r, ns, done)

    def calc_advantage(self, s, r, ns, done):

        with torch.no_grad():
            td_target = r + self.gamma * self.network.value(ns) * (1-done)
            delta = td_target - self.network.value(s)

        adv = 0.0
        Advantage = torch.zeros_like(delta)

        for i in reversed(range(len(Advantage))):
            adv = delta[i] + self.rambda * self.gamma * adv
            Advantage[i] = adv

        Advantage = (Advantage - Advantage.mean()) / (Advantage.std() + 1e-7)

        return td_target, Advantage


    def update(self):

        s, a, r, ns, done = self.memory.get_sample()

        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)
        r = torch.cat(r, dim=0)
        ns = torch.cat(ns, dim=0)
        done = torch.cat(done, dim=0)

        td_target, Advantage = self.calc_advantage(s, r, ns, done)

        with torch.no_grad():
            old_dist = self.network.policy(s)
            old_log_prob = old_dist.log_prob(a)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SequentialSampler(range(len(s))), 8, drop_last=False):

                dist = self.network.policy(s[index])
                new_log_prob = dist.log_prob(a[index])
                entropy = dist.entropy()

                ratio = torch.exp(new_log_prob - old_log_prob[index])
                surr1 = ratio * Advantage[index]
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * Advantage[index]
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.network.value(s[index])
                critic_loss = self.criteria(value, td_target[index])

                loss = actor_loss + critic_loss - 0.01*entropy
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()


        self.memory.reset()


def main():

    env = gym.make('Pendulum-v0')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = PPO(s_dim, a_dim, max_action)

    mean_array = []

    for ep in range(2000):
        cum_r = 0
        s = env.reset()
        while True:        
            a = agent.get_action(s)
            ns, r, done, info = env.step(a)
            agent.get_sample(s, a, r/10., ns, done)

            s = ns
            cum_r += r

            if done:
                break

        mean_array.append(cum_r)
        agent.update()

        if ep % 20 == 0 and ep != 0:
            print('{}th episode {} mean reward'.format(ep, np.mean(mean_array)))
            mean_array = []


if __name__ == "__main__":
    main()


    