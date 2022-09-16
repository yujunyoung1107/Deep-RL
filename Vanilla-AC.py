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


class ActorCritic(nn.Module):
    
    def __init__(self, s_dim, a_dim, device):
        super(ActorCritic, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor = MLP(s_dim, a_dim, num_neurons=[256]).to(device)
        self.critic = MLP(s_dim, 1, num_neurons=[256]).to(device)
        self.softmax = nn.Softmax(dim=-1)
        self.logmax = nn.LogSoftmax(dim=-1)
        self.parameters = list(self.actor.parameters()) + list(self.critic.parameters())

        self.optimizer = torch.optim.Adam(self.parameters, lr=3e-4)
        self.criteria = nn.SmoothL1Loss()
        self.memory = Buffer()
        self.gamma = 0.99
        self.device = device


    def get_action(self, state):

        state = ToTensor(state).to(self.device)
        prob = self.softmax(self.actor(state))
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

        s, a, r, ns, done = self.memory.get_sample()

        s = torch.cat(s, dim=0).float().to(self.device)
        a = torch.cat(a, dim=0).to(self.device)
        r = torch.cat(r, dim=0).float().to(self.device)
        ns = torch.cat(ns, dim=0).float().to(self.device)
        done = torch.cat(done, dim=0).float().to(self.device)

        log_prob = self.logmax(self.actor(s)).gather(1,a)
        value = self.critic(s)

        with torch.no_grad():
            td_target = r + self.gamma*self.critic(ns)*(1-done)
            td_error = td_target - value

        critic_loss = (value - td_target).pow(2)
        actor_loss = -(log_prob * td_error)
        total_loss = (critic_loss + actor_loss).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
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
                    self.update()
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

    agent = ActorCritic(s_dim, a_dim, device)

    agent.run_episode(env, 1000)
        

if __name__ == "__main__":
    main()
