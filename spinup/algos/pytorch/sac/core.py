import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from gym.spaces import Box, Discrete

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def flattened_shape(shape):
    flat_dim = 1
    for d in shape:
        flat_dim*=d
    return flat_dim

def flatten(a):
    batch_size = a.shape[0]
    if torch.is_tensor(a):
        return a.view(batch_size,-1)
    else:
        return a.reshape(batch_size,-1)

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


class MLPCategoricalActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.probs_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation = nn.Softmax)

    def forward(self, obs, deterministic=False, with_logprob=True):
        # obs = flatten(obs)
        probs = self.probs_net(obs)
        pi_distribution = Categorical(probs=probs)

        if deterministic:
            pi_action = torch.argmax(probs, dim=-1)
        else:
            pi_action = pi_distribution.sample()

        if with_logprob:
            log_pi = pi.log_prob(pi_action)
        else:
            log_pi = None

        return pi_action, log_pi

    def get_probs(self, obs):
        # obs = flatten(obs)
        probs = self.probs_net(obs)
        log_probs = torch.log(probs+1e-10)
        return probs, log_probs


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPCategoricalQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, act = None):
        q = self.q(obs)
        if not act is None:
            bSize = q.shape[0]
            q = q[torch.arange(bSize), act]
        return q

        
class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]

            self.is_discrete = False
            # build policy and value functions
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        elif isinstance(action_space, Discrete):
            self.is_discrete = True
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
            self.q1 = MLPCategoricalQFunction(obs_dim, action_space.n, hidden_sizes, activation)
            self.q2 = MLPCategoricalQFunction(obs_dim, action_space.n, hidden_sizes, activation)
            
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

    def get_probs(self, obs):
        assert self.is_discrete, "Action space needs to be discrete"
        return self.pi.get_probs(obs)


    