import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical
from Network.MLP import MLP
from Memory.Memory import ReplayBuffer
from Utils.utils import *

class Actor(nn.Module):

    def __init__(self, s_dim, a_dim, max_act):
        super(Actor, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.max_act = max_act

        self.network = MLP(s_dim, a_dim, num_neurons=[128,128], out_act='Tanh')

    def forward(self, state):

        action = self.network(state)
        return action * self.max_act

    
class Critic(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.network = MLP(s_dim + a_dim, 1, num_neurons=[128,128], out_act='Identity')

    def forward(self, state, action):

        return self.network(torch.cat([state, action], dim=-1))


class DDPG(nn.Module):

    def __init__(self, s_dim, a_dim, max_act):
        super(DDPG, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.max_act = max_act
        
        self.actor = Actor(s_dim, a_dim, max_act)
        self.actor_target = Actor(s_dim, a_dim, max_act)
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.criteria = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(50000)
        self.gamma = 0.99
        self.tau = 0.005

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def get_action(self, state):

        state = ToTensor(state)
        action = self.actor(state)
        
        return action.view(-1).detach().numpy()

    def get_sample(self, s, a, r, ns, done):

        s = ToTensor(s)
        a = ToTensor(a)
        r = ToTensor(r)
        ns = ToTensor(ns)
        done = ToTensor(done)

        self.memory.push(s, a, r, ns, done)

    def update(self):

        s, a, r, ns, done = self.memory.get_sample(256)

        with torch.no_grad():
            target = r + self.gamma*self.critic_target(ns, self.actor_target(ns)) * (1-done)

        critic_loss = self.criteria(self.critic(s,a), target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


    def soft_target_update(self):
        for act_p, act_tar_p, cri_p, cri_tar_p  in zip(self.actor.parameters(), self.actor_target.parameters(), self.critic.parameters(), self.critic_target.parameters()):
            act_tar_p.data.copy_(act_tar_p.data*(1.0-self.tau) + act_p.data*self.tau)
            cri_tar_p.data.copy_(cri_tar_p.data*(1.0-self.tau) + cri_p.data*self.tau)
        
        

def main():

    env = gym.make('Pendulum-v0')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = DDPG(s_dim, a_dim, max_action)

    for ep in range(2000):
        ou_noise = OUProcess(mu=np.zeros(1))
        cum_r = 0
        s = env.reset()
        while True:        
            a = agent.get_action(s) + ou_noise()[0]
            ns, r, done, info = env.step(a)
            agent.get_sample(s, a, r, ns, done)

            s = ns
            cum_r += r

            if len(agent.memory) > 2000:
                agent.update()
                agent.soft_target_update()

            if done:
                break

        print('episode : {} | reward : {}'.format(ep, cum_r))
        

if __name__ == "__main__":
    main()



