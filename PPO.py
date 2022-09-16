from stat import S_IMODE
import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from Network.MLP import MLP
from Memory.Memory import Buffer
from Utils.utils import *

class Network(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Network, self).__init__()

        self.actor = MLP(s_dim, a_dim, [256,256])
        self.critic = MLP(s_dim, 1, [256,256])
        self.softmax = nn.Softmax(dim=-1)
        self.logmax = nn.LogSoftmax(dim=-1)

    def forward(self, state):
        
        x = self.actor(state)
        prob = self.softmax(x)
        log_prob = self.logmax(x)
        value = self.critic(state)

        return prob, log_prob, value


class PPO(nn.Module):
    
    def __init__(self, s_dim, a_dim):
        super(PPO, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = Network(s_dim, a_dim)
        self.opt = torch.optim.Adam(params=self.net.parameters(), lr=3e-5)
        self.memory = Buffer()
        self.gamma = 0.99
        self.rambda = 0.9
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.batch_size = 32

    def get_action(self, state):

        state = torch.tensor(state).view(1,-1).float()

        with torch.no_grad():
            prob, _, _ = self.net(state)
            action = Categorical(prob).sample()

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
            td_target = r + self.gamma * self.net.critic(ns) * (1-done)
            delta = td_target - self.net.critic(s)

        adv = 0.0
        Advantage = torch.zeros_like(delta)

        for i in reversed(range(len(Advantage))):
            adv = delta[i] + self.rambda * self.gamma * adv
            Advantage[i] = adv

        Advantage = (Advantage - Advantage.mean()) / Advantage.std()

        return td_target, Advantage


    def update(self):

        s, a, r, ns, done = self.memory.get_sample()

        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)
        r = torch.cat(r, dim=0)
        ns = torch.cat(ns, dim=0)
        done = torch.cat(done, dim=0)

        td_target, Advantage = self.calc_advantage(s, r, ns, done)

        old_prob, old_log_prob, old_value = self.net(s)
        old_log_prob = old_log_prob.gather(1,a).detach()


        for pe in range(self.ppo_epoch):
            for index in BatchSampler(SequentialSampler(range(len(s))), 32, drop_last=False):
                prob, log_prob, value = self.net(s[index])
                entropy = -(prob*log_prob).sum(-1)
                new_log_prob = log_prob.gather(1, a[index])

                ratio = torch.exp(new_log_prob - old_log_prob[index])
                surr1 = ratio * Advantage[index]
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * Advantage[index]

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = (td_target[index] - value).pow(2)
                
                loss = (actor_loss + 0.5*critic_loss - 0.01*entropy).mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5)
                self.opt.step()

        self.memory.reset()


def main():


    env = gym.make('LunarLander-v2')
    #env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = PPO(s_dim, a_dim)

    for ep in range(2000):
        cum_r = 0
        s = env.reset()[0]
        while True:       
            env.render() 
            a = agent.get_action(s)
            ns, r, done, trunc, _ = env.step(a.item())
            if done == False and trunc == False:
                new_done = False
            else:
                new_done = True
            agent.get_sample(s, a, r/100, ns, new_done)
            s = ns
            cum_r += r

            if new_done:
                if len(agent.memory) > 100:
                    agent.update()
                break

        print('episode : {} | reward : {}'.format(ep, cum_r))
        

if __name__ == "__main__":
    main()


     