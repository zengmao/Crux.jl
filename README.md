# Note

This is a fork of [Anthony Corso's Crux.jl package](https://github.com/sisl/Crux.jl) for deep reinforcement learning in Julia, with some broken dependencies (particularly interfaces with OpenAI Gym) stripped off, so that much of the rest of the package remains available and installable with the latest versions of Julia before possible attempts to fix the original package. Most of the examples and tests dependent on the Python OpenAI Gym environments are therefore deleted. However, the core package for solving custom RL environments written in the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface remains working.

Currently, the package works with Julia 1.11 under Windows, Linux, and MacOS, and works with Julia 1.10 under Linux.

In <a href="./examples/rl/cartpole.jl">examples/rl/cartpole.jl</a>, we use the CartPole environment provided by `ReinforcementLearningEnvironments.jl` and convert it into the POMDPs interface, as a replacement of the OpenAI Gym equivalent of this environment. To try this example, first include the package of this repo in the Julia REPL (in a new virtual environment):
```
]add https://github.com/zengmao/Crux.jl.git
```
Then install additional packages needed for the script:
```
]add POMDPs, QuickPOMDPs, POMDPTools, ReinforcementLearningEnvironments, Random, Flux, Plots
```
Then download a copy of <a href="./examples/rl/cartpole.jl">examples/rl/cartpole.jl</a> and run it. The code solves the environment with several RL algorithms and plots the learning curves in `cartpole_training.pdf`. The PPO training outcome will be shown as an animation. Log files will be written to `logs/`.

In case any dependency becomes broken in the future, please switch to the `manifest` branch which contains the exact last working versions of dependencies, in `Manifest-v1.11.toml` for Julia 1.11 and `Manifest.toml` for Julia 1.10, on Linux (x86_64 glibc).

Below is the original README.

# Crux.jl

[![Build Status](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml)
[![Code Coverage](https://codecov.io/gh/sisl/Crux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/Crux.jl)

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements the following algorithms:
### Reinforcement Learning
* <a href="./src/model_free/rl/dqn.jl">Deep Q-Learning</a>
  * Prioritized Experience Replay
* <a href="./src/model_free/rl/softq.jl">Soft Q-Learning</a>
* <a href="./src/model_free/rl/reinforce.jl">REINFORCE</a>
* <a href="./src/model_free/rl/ppo.jl">Proximal Policy Optimization (PPO)</a>
* Lagrange-Constrained PPO
* <a href="./src/model_free/rl/a2c.jl">Advantage Actor Critic</a>
* <a href="./src/model_free/rl/ddpg.jl">Deep Deterministic Policy Gradient (DDPG)</a>
* <a href="./src/model_free/rl/td3.jl">Twin Delayed DDPG (TD3)</a>
* <a href="./src/model_free/rl/sac.jl">Soft Actor Critic (SAC)</a>

### Imitation Learning
* <a href="./src/model_free/il/bc.jl"> Behavioral Cloning </a>
* <a href="./src/model_free/il/gail.jl">Generative Adversarial Imitation Learning (GAIL) w/ On-Policy and Off Policy Versions</a>
* <a href="./src/model_free/il/AdVIL.jl">Adversarial value moment imitation learning (AdVIL)</a>
* <a href="./src/model_free/il/AdRIL.jl">(AdRIL)</a>
* <a href="./src/model_free/il/sqil.jl">(SQIL)</a>
* <a href="./src/model_free/il/asaf.jl">Adversarial Soft Advantage Fitting (ASAF)</a>
* <a href="./src/model_free/il/iqlearn.jl">Inverse Q-Learning (IQLearn)</a>

### Batch RL
* <a href="./src/model_free/batch/sac.jl">Batch Soft Actor Critic (SAC)</a>
* <a href="./src/model_free/batch/cql.jl">Conservative Q-Learning (CQL)</a>

### Adversarial RL
* <a href="./src/model_free/adversarial/rarl.jl">Robust Adversarial RL (RARL)</a>

### Continual Learning
* Experience Replay


## Installation

* Install <a href="https://github.com/ancorso/POMDPGym">POMDPGym</a>
* Install by opening julia and running `] add Crux`

To edit or contribute use `] dev Crux` and the repo will be cloned to `~/.julia/dev/Crux`

Maintained by Anthony Corso (acorso@stanford.edu)
