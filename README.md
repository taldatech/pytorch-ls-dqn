# pytorch-ls-dqn
PyTorch implementation of Least-Squares DQN with extras (DuelingDQN, Boosted FQI)

Based on the paper:

Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://arxiv.org/abs/1705.07461), NIPS 2017

Video:

YouTube - not yet available


![pong](https://github.com/taldatech/pytorch-ls-dqn/blob/master/images/pong.gif)
![boxing](https://github.com/taldatech/pytorch-ls-dqn/blob/master/images/boxing.gif)

## Background



## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`|  `0.4.1`|
|`gym`|  `0.10.9`|
|`tensorboard`|  `1.12.0`|
|`tensorboardX`|  `1.5`|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`ls_dqn_main.py`| general purpose main application for training/playing a LS-DQN agent|
|`pong_ls_dqn.py`| main application tailored for Atari's Pong|
|`boxing_ls_dqn.py`| main application tailored for Atari's Boxing|
|`dqn_play.py`| sample code for playing a game, also in `ls_dqn_main.py`|
|`actions.py`| classes for actions selection (argmax, epsilon greedy)|
|`agent.py`| agent class, holds the network, action selector and current state|
|`dqn_model.py`| DQN classes, neural networks structures|
|`experience.py`| Replay Buffer classes|
|`hyperparameters.py`| hyperparameters for several Atari games, used as a baseline|
|`srl_algorithms.py`| Shallow RL algorithms, LS-UPDATE|
|`utils.py`| utility functions|
|`wrappers.py`| DeepMind's wrappers for the Atari environments|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|


## API (`ls_dqn_main.py --help`)


You should use the `ls_dqn_main.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading |
|-k, --lsdqn	| use LS-DQN (apply LS-UPDATE every N_DRL), default: false |
|-j, --boosting| use Boosted-FQI as SRL algorithm, default: false |
|-u, --double| use double dqn, default: false|
|-f, --dueling| use dueling dqn, default: false |
|-m, --cond_update| conditional ls-update: update only if ls weights are better, default: false |
|-e, --env| environment to play: pong, boxing, breakout, breakout-small, invaders |
|-d, --decay_rate| number of episodes for epsilon decaying, default: 100000 |
|-o, --optimizer| optimizing algorithm ('RMSprop', 'Adam'), deafult: 'Adam' |
|-r, --learn_rate| learning rate for the optimizer, default: 0.0001 |
|-g, --gamma| gamma parameter for the Q-Learning, default: 0.99 |
|-l, --lam| regularization parameter value, default: 1, 10000 (boosting) |
|-s, --buffer_size| Replay Buffer size, default: 1000000 |
|-b, --batch_size| number of samples in each batch, default: 128 |
|-i, --steps_to_start_learn| number of steps before the agents starts learning, default: 10000 |
|-c, --target_update_freq| number of steps between copying the weights to the target DQN, default: 10000 |
|-x, --record| Directory to store video recording when playing (only Linux) |
|--no-visualize| if typed, render the environment when playing. default: True |

## Playing
Agents checkpoints (files ending with `.pth`) are saved and loaded from the `agent_ckpt` directory.
Playing a pretrained agent for one episode:

`python ls_dqn_main.py --play -y ./agent_ckpt/pong_agent.pth --no-visualize`

If the checkpoint was trained using Dueling DQN:

`python ls_dqn_main.py --play -f -y ./agent_ckpt/pong_agent.pth --no-visualize`

## Training

Examples:

* `python ls_dqn_main.py --train --lsdqn -e boxing -l 10 -b 64`
* `python ls_dqn_main.py --train --lsdqn --boosting --dueling -m -e boxing -l 1000 -b 64`

For full description of the flags, see the full API.

## Playing Atari on Windows

You can train and play on a Windows machine, thanks to Nikita Kniazev, as follows from this post on [stackoverflow](https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299)

`pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py` 

## References
* [PyTorch Agent Net: reinforcement learning toolkit for pytorch](https://github.com/Shmuma/ptan) by [Max Lapan](https://github.com/Shmuma)
* Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://arxiv.org/abs/1705.07461), NIPS 2017



