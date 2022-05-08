import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from Network.MLP import MLP
from Memory.Memory import ReplayBuffer
from Utils.utils import *

class DQN(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(DQN, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.Qnet = MLP(s_dim, a_dim, num_neurons=[256,256])
        self.Qnet_target = MLP(s_dim, a_dim, num_neurons=[256,256])

        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=3e-4)
        self.criteria = nn.MSELoss()
        self.memory = ReplayBuffer(50000)
        self.gamma = 0.99
        self.eps = 0.08

        self.Qnet_target.load_state_dict(self.Qnet.state_dict())

    
    def get_action(self, state):

        state = torch.tensor(state).view(-1,self.s_dim).float()
        action_prob = np.random.uniform(0.0, 1.0, 1)
        if action_prob > self.eps:
            action = self.Qnet(state).argmax(-1)
        else:
            action = torch.randint(0, self.a_dim, (1,))

        return action.view(-1).detach().numpy()


    def get_sample(self, s, a, r, ns, done):

        s = ToTensor(s)
        a = ToTensor(a, False)
        r = ToTensor(r)
        ns = ToTensor(ns)
        done = ToTensor(done)

        self.memory.push(s, a, r, ns, done)

    
    def update(self):

        s, a, r, ns, done = self.memory.get_sample(256)

        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)
        r = torch.cat(r, dim=0)
        ns = torch.cat(ns, dim=0)
        done = torch.cat(done, dim=0)

        with torch.no_grad():
            Qmax, _ = self.Qnet_target(ns).max(dim=-1, keepdim=True)
            Q_target = r + self.gamma*Qmax*(1-done)

        Qvalue = self.Qnet(s).gather(1,a)
        Loss = self.criteria(Qvalue, Q_target)

        self.optimizer.zero_grad()
        Loss.backward()
        self.optimizer.step()
    
    def target_update(self):

        self.Qnet_target.load_state_dict(self.Qnet.state_dict())



def main():

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(s_dim, a_dim)

    for ep in range(2000):

        eps = max(0.05, 0.08 - 0.03 * (ep / 200))
        agent.eps = eps
        cum_r = 0
        s = env.reset()

        while True:        
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())
            agent.get_sample(s, a, r/100, ns, done)
            s = ns
            cum_r += r

            if len(agent.memory) > 2000:
                agent.update()

            if done:
                break

        print('episode : {} | reward : {}'.format(ep, cum_r))

        if ep % 10 == 0:
            agent.target_update()
        

if __name__ == "__main__":
    main()
