**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


## Important points
1. VPG:
	1. Uses GAE for advantage estimate with discounting (gamma & lambda). But expectation doesn't assume discounting
	1. Normalizes advantage estimates over one epoch (0 mean, 1 std)
	1. V used at time t actually corresponds to policy at time t-1. V always lags behind. So probably we don't want the policy to change too much :(
	1. I tried rev, updating value before pi but no improvement
	1. While training value function, target is fixed

1. PPO: 
	1. Updates pi and v separately, no entropy bonus
	1. Early stoppping if KL(pi_old||pi) crosses threshold
	1. v lags behind pi as in VPG

## Getting Results
1. Plotting command looks like:
`python -m spinup.run plot /home/ankur/MSR_Research_Home/spinningup/data/cmd_ppo_pytorch/cmd_ppo_pytorch_s0`

1. Watch the trained agent with a command like:
`python -m spinup.run test_policy /home/ankur/MSR_Research_Home/spinningup/data/cmd_ppo_pytorch/cmd_ppo_pytorch_s0`

Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```