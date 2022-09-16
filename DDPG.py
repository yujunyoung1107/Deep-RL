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

    def __init__(self, s_dim, a_dim, max_act, device):
        super(DDPG, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.max_act = max_act
        
        self.actor = Actor(s_dim, a_dim, max_act).to(device)
        self.actor_target = Actor(s_dim, a_dim, max_act).to(device)
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.criteria = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(50000, device)
        self.gamma = 0.99
        self.tau = 0.005
        self.device = device

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def get_action(self, state):

        state = ToTensor(state).to(self.device)
        action = self.actor(state)
        
        return action.view(-1).detach().cpu().numpy()

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

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

    def run_episode(self, env, num_episode):

        reward_sum = []

        for ep in range(num_episode):
            s = env.reset()[0]
            cum_r = 0
            ou_noise = OUProcess(mu=np.zeros(1))
            while True:
                a = self.get_action(s) + ou_noise()[0]
                ns, r, done, trunc, _ = env.step(a)
                new_done = False if (done==False and trunc==False) else True

                self.get_sample(s, a, r, ns, new_done)

                s = ns
                cum_r += r

                if len(self.memory) > 2000:
                    self.update()

                if new_done:
                    reward_sum.append(cum_r)
                    break

            print('ep : {} | reward : {}'.format(ep, cum_r))

            if ep % 10 == 0 and ep != 0:
                print('ep : {} | reward_avg : {}'.format(ep, np.mean(reward_sum)))
                reward_sum = []



        
        

def main():

    env = gym.make('Pendulum-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    agent = DDPG(s_dim, a_dim, max_action, device)

    agent.run_episode(env,500)

        
        

if __name__ == "__main__":
    main()



