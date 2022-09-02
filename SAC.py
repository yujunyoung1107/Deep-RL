import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import gym
from Network.MLP import MLP
from Memory.Memory import ReplayBuffer
from Utils.utils import *

class Qnet(nn.Module):
    
    def __init__(self, s_dim, a_dim , lr):
        super(Qnet, self).__init__()

        self.network = MLP(input_dim=s_dim+a_dim, output_dim=1, num_neurons=[256,256])
        self.opt = torch.optim.Adam(params=self.network.parameters(), lr=lr)
        self.apply(init_weights)
    
    def forward(self, state, action):
        return self.network(torch.cat([state,action], dim=1))


class Actor(nn.Module):

    def __init__(self, s_dim, a_dim, lr, max_action):
        super(Actor, self).__init__()

        self.network = MLP(input_dim=s_dim, output_dim=256, num_neurons=[256,256], out_act='ReLU')
        self.mu = nn.Linear(256, a_dim)
        self.sigma = nn.Linear(256, a_dim)
        self.reparam_noise = 1e-6
        self.max_action = max_action
        self.opt = torch.optim.Adam(params=self.parameters(), lr=lr)
        self.apply(init_weights)

    def forward(self, state):

        xs = self.network(state)
        mu = self.mu(xs)
        sigma = self.sigma(xs)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1.0)
        
        dist = Normal(mu, sigma)
        a_sample = dist.rsample()     
        action = torch.tanh(a_sample)
        log_prob = dist.log_prob(a_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + self.reparam_noise)

        return action * self.max_action, log_prob


class SAC(nn.Module):

    def __init__(self, s_dim, a_dim, max_action):
        super(SAC, self).__init__()

        self.actor = Actor(s_dim, a_dim, 3e-4, max_action)
        self.Q1 = Qnet(s_dim, a_dim, 3e-4)
        self.Q2 = Qnet(s_dim, a_dim, 3e-4)
        self.Q1_target = Qnet(s_dim, a_dim, 3e-4)
        self.Q2_target = Qnet(s_dim, a_dim, 3e-4)
        self.log_alpha = torch.zeros(1)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3) 

        self.Q1_target.load_state_dict(self.Q1.state_dict())    
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.memory = ReplayBuffer(1000000)
        self.gamma = 0.99
        self.criterion = nn.MSELoss()
        self.target_entropy = -float(a_dim)
        self.tau = 0.005

    def get_action(self, state):

        state = ToTensor(state)
        action, log_prob = self.actor(state)
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
            action, log_prob = self.actor(ns)
            Q1 = self.Q1_target(ns, action)
            Q2 = self.Q2_target(ns, action)

            Q_value = torch.min(Q1, Q2)
            Q_target = r + self.gamma * (1-done) * (Q_value - log_prob.mean(1, keepdim=True) * self.log_alpha.exp())
        
        Q1_loss = self.criterion(self.Q1(s,a), Q_target)
        Q2_loss = self.criterion(self.Q2(s,a), Q_target)

        self.Q1.opt.zero_grad()
        self.Q2.opt.zero_grad()

        Q_loss = ((Q1_loss + Q2_loss) / 2).mean()
        Q_loss.backward()

        self.Q1.opt.step()
        self.Q2.opt.step()

        action, log_prob = self.actor(s)    
        Q_value = torch.min(self.Q1(s, action), self.Q2(s, action)) 
        actor_loss = (self.log_alpha.exp() * log_prob - Q_value).mean()
        self.actor.opt.zero_grad()
        actor_loss.backward()
        self.actor.opt.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        soft_update(self.Q1, self.Q1_target, self.tau)
        soft_update(self.Q2, self.Q2_target, self.tau)

    
    def run_episode(self, env, num_episode):

        reward_sum = 0.

        for ep in range(num_episode):
            s = env.reset()
            cum_r = 0
            while True:
                a = self.get_action(s)
                ns, r, done, _ = env.step(a)
                self.get_sample(s,a,r,ns,done)

                if len(self.memory) > 1e3:
                    self.update()

                s = ns
                cum_r += r 

                if done:
                    print(cum_r)
                    reward_sum += cum_r
                    break

            if ep % 10 == 0 and ep != 0:
                print('ep : {} | reward_avg : {}'.format(ep, reward_sum))
                reward_sum = 0.


def main():

    env = gym.make('BipedalWalker-v3')
    s_dim = env.observation_space.shape[0]  
    a_dim = env.action_space.shape[0]
    high = env.action_space.high[0]

    agent = SAC(s_dim, a_dim, high)
    agent.run_episode(env, 500)


if __name__ == '__main__':
    main()
