# pytorch-ls-dqn
PyTorch implementation of Least-Squares DQN

Based on the paper:
Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://arxiv.org/abs/1705.07461), NIPS 2017

Video:

YouTube - 


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
|`taxi_main.py`| Main application for training/playing a Taxi-v2 agent|
|`acrobot_main.py`| Main application for training/playing a Acrobot-v1 agent|
|`Agent.py`| Classes: TaxiAgent, AcrobotAgent|
|`DQN_model.py`| Classes: DQN_DNN, DQN_CNN (the neural networks architecture)|
|`Helpers.py`| Helper functions (converting states, plotting...)|
|`OneHotGenerator.py`| Class: OneHotGenerator (converts integers to One-Hot-Vectors)|
|`ReplayBuffer.py`| Class: ReplayBuffer (stores memories for the DQN learning)|
|`Schedule.py`| Classes: ExponentialSchedule, LinearSchedule (scheduling of epsilon-greedy policy)|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|
|`*_training.status`| Pickle files with the recent training status for a model (episodes seen, total rewards...)|
|`Taxi_Agent.ipynb` | Jupyter Notebook with detailed explanation, derivations and graphs for the Taxi-v2 environemnt| 
|`Acrobot_Agent.ipynb` | Jupyter Notebook with detailed explanation, derivations and graphs for the Acrobot-v1 environemnt| 
|`dqn_pg_writeup_gh.pdf` | Summary of this work| 


## Taxi-v2 Environment DQN

### API (`python taxi_main.py --help`)


You should use the `taxi_main.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading, if not set, training will continue from a pretrained checkpoint |
|-m, --mode	| model's mode or state representation ('one-hot', 'location-one-hot'), default: 'one-hot' |
|-e, --episodes| number of episodes to play or train, default: 2 (play), 5000 (train) |
|-x, --exploration| epsilon-greedy scheduling ('exp', 'lin'), default: 'exp'|
|-d, --decay_rate| number of episodes for epsilon decaying, default: 800 |
|-u, --hidden_units| number of neurons in the hidden layer of the DQN, default: 150 |
|-o, --optimizer| optimizing algorithm ('RMSprop', 'Adam'), deafult: 'RMSProp' |
|-r, --learn_rate| learning rate for the optimizer, default: 0.0003 |
|-g, --gamma| gamma parameter for the Q-Learning, default: 0.99 |
|-s, --buffer_size| Replay Buffer size, default: 500000 |
|-b, --batch_size| number of samples in each batch, default: 128 |
|-i, --steps_to_start_learn| number of steps before the agents starts learning, default: 1000 |
|-c, --target_update_freq| number of steps between copying the weights to the target DQN, default: 5000 |
|-a, --clip_grads| use Gradient Clipping regularization (default: False) |
|-z, --batch_norm| use Batch Normalization between DQN's layers (default: False) |
|-y, --dropout| use Dropout regularization on the layers of the DQN (default: False) |
|-q, --dropout_rate| probability for a layer to be dropped when using Dropout, default: 0.4 |

### Playing
Agents checkpoints (files ending with `.pth`) are saved and loaded from the `taxi_agent_ckpt` directory.
Playing a pretrained agent for 2 episodes:

`python taxi_main.py --play`

For more more episodes, e.g 10:

`python taxi_main.py --play -e 10`

Playing a pretrained agent with Location-One-Hot state representation:

`python taxi_main.py --play -m location-one-hot -e 3`

For playing another chekpoint, the `-n` flag must correspond with a `.pth` checkpoint file in the `taxi_agent_ckpt` directory.

### Training

Note: in order to continue training from a pretrained checkpoint you can either:

	1. Name the model with the same name as the saved chekpoint (e.g. if the there exists `taxi_agent_user.pth` the model name should be `user`)
	
	2. Leave out the name (don't use the `-n` flag) and a default pretrained checkpoint will be loaded and a random name will be given (which you can change later)

Examples:

* `python taxi_main.py --train -n my_taxi -m one-hot -e 5000 -x exp -d 800 -u 150 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 128 -i 2000 -c 5000`
* `python taxi_main.py --train -a -m location-one-hot -e 5000 -x lin -d 800 -u 150 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 128 -i 2000 -c 5000`

For full description of the flags, see the full API.

## Playing Atari on Windows

## References
* PTAN
* Original Paper



