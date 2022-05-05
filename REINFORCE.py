import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical
from Network.MLP import MLP
from Memory.Buffer import Buffer

def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE")
    parser.add_argument('--update', dest='update', help='vanilla / episodic',
                        default='vanilla', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

class REINFORCE(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(REINFORCE, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.policy = MLP(s_dim, a_dim, num_neurons=[256])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.softmax = nn.Softmax(dim=-1)
        self.logmax = nn.LogSoftmax(dim=-1)
        self.memory = Buffer()
        self.gamma = 0.99


    def get_action(self, state):

        state = torch.tensor(state).view(-1, self.s_dim).float()
        prob = self.softmax(self.policy(state))
        action = Categorical(prob).sample()

        return action.view(-1).detach().numpy()

    def get_sample(self, s, a, r):
        
        s = torch.tensor(s).view(1, self.s_dim).float()
        a = torch.tensor(a).view(1,1)
        r = torch.tensor(r).view(1,1).float()
        
        self.memory.push(s, a, r)

    def update(self):
        
        states, actions, rewards = self.memory.get_sample()

        G = 0.0
        for s, a, r in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = r + self.gamma*G

            log_prob = self.logmax(self.policy(s)).view(-1)[a.item()]
            Loss = -log_prob*G

            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()

        self.memory.reset()

    def episodic_update(self):

        states, actions, rewards = self.memory.get_sample()

        G = 0.0
        Reward2Go = []
        for r in rewards[::-1]:
            G = r + self.gamma*G
            Reward2Go.insert(0,G)

        Reward2Go = torch.tensor(Reward2Go).float()
        Reward2Go = (Reward2Go - Reward2Go.mean()) / Reward2Go.std()
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)

        log_prob = self.logmax(self.policy(states))
        log_prob = log_prob.gather(1, actions).view(-1)
        Loss = -(log_prob * Reward2Go).mean()

        self.optimizer.zero_grad()
        Loss.backward()
        self.optimizer.step()
        
        self.memory.reset()


def main():

    args = parse_args()

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = REINFORCE(s_dim, a_dim)

    for ep in range(2000):
        cum_r = 0
        s = env.reset()

        while True:
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())
            agent.get_sample(s, a, r)
            s = ns
            cum_r += r

            if done:
                break

        if args.update == 'vanilla':
            agent.update()
        else:
            agent.episodic_update()

        print('episode : {} | reward : {}'.format(ep, cum_r))
        

if __name__ == "__main__":
    main()