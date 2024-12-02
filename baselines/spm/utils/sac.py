from copy import deepcopy
import itertools

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256), alpha=0.2,
                activation=nn.ReLU):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        self.alpha = alpha

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
    
    
class SACAgent(nn.Module):
    
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256, 256), alpha=0.2,
                gamma=0.99, lr=1e-3, polyak=0.995,
                activation=nn.ReLU):
        
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = MLPActorCritic(obs_dim, act_dim, act_limit)
        self.ac.to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)
        
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
    
        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak
        
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr)  
        
        self.lr = lr
        
    def reset(self):
        
        self.ac = MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit)
        self.ac.to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)
        
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
    
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=self.lr)  
    
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, r, o2, d = o.to(self.device), a.to(self.device), r.to(self.device), o2.to(self.device), d.to(self.device)
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)
        
        with torch.no_grad():
            
            a2, logp_a2 = self.ac.pi(o2)
            
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
            
            
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                    Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info
    
    def compute_loss_pi(self, data):
        o = data['obs']
        o = o.to(self.device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info
    
    def update(self, data):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        
        for p in self.q_params:
            p.requires_grad = False
        
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        
        for p in self.q_params:
            p.requires_grad = True
        
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.ac.pi(obs, deterministic, False)
            return a.cpu().numpy()
        
    def get_action(self, o):
        return self.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic=True)