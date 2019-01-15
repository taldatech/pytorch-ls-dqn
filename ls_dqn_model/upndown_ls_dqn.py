import gym
# import argparse
import torch
import torch.optim as optim
import os

from tensorboardX import SummaryWriter

import ls_dqn_model.utils.dqn_model as dqn_model
from ls_dqn_model.utils.hyperparameters import HYPERPARAMS
from ls_dqn_model.utils.agent import DQNAgent, TargetNet
from ls_dqn_model.utils.actions import EpsilonGreedyActionSelector
import ls_dqn_model.utils.experience as experience
import ls_dqn_model.utils.utils as utils
from ls_dqn_model.utils.srl_algorithms import ls_step
import ls_dqn_model.utils.wrappers as wrappers
import numpy as np
import random
import copy


if __name__ == "__main__":
    params = HYPERPARAMS['upndown']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(params['env_name'])
    env = wrappers.wrap_dqn(env)

    training_random_seed = 10
    save_freq = 50000
    n_drl = 500000  # steps of DRL between SRL
    n_srl = params['replay_size']  # size of batch in SRL step
    num_srl_updates = 3  # number of to SRL updates to perform
    use_double_dqn = False
    use_dueling_dqn = True
    use_boosting = False
    use_ls_dqn = False
    use_constant_seed = False  # to compare performance independently of the randomness
    save_for_analysis = False  # save also the replay buffer for later analysis

    lam = 10  # regularization parameter
    # params['batch_size'] = 64
    if use_ls_dqn:
        print("using ls-dqn with lambda:", str(lam))
        model_name = "-LSDQN-LAM-" + str(lam) + "-" + str(int(1.0 * n_drl / 1000)) + "K"
    else:
        model_name = "-DQN"
    model_name += "-BATCH-" + str(params['batch_size'])
    if use_double_dqn:
        print("using double-dqn")
        model_name += "-DOUBLE"
    if use_dueling_dqn:
        print("using dueling-dqn")
        model_name += "-DUELING"
    if use_boosting:
        print("using boosting")
        model_name += "-BOOSTING"
    if use_constant_seed:
        model_name += "-SEED-" + str(training_random_seed)
        np.random.seed(training_random_seed)
        random.seed(training_random_seed)
        env.seed(training_random_seed)
        torch.manual_seed(training_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_random_seed)
        print("training using constant seed of ", training_random_seed)

    writer = SummaryWriter(comment="-" + params['run_name'] + model_name)
    if use_dueling_dqn:
        net = dqn_model.DuelingLSDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = utils.EpsilonTracker(selector, params)

    agent = DQNAgent(net, selector, device=device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])  # TODO: change to RMSprop

    utils.load_agent_state(net, optimizer, selector, load_optimizer=False, env_name='boxing',
                           path='./agent_ckpt/agent_ls_dqn_-upndown.pth')

    frame_idx = 0
    drl_updates = 0

    with utils.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    if save_for_analysis:
                        temp_model_name = model_name + "_" + str(frame_idx)
                        utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                               selector.epsilon, save_replay=True, replay_buffer=buffer.buffer,
                                               name=temp_model_name)
                    else:
                        utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                               selector.epsilon, name='-upndown')
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                         device=device, double_dqn=use_double_dqn)
            loss_v.backward()
            optimizer.step()
            drl_updates += 1

            # LS-UPDATE STEP
            if use_ls_dqn and (drl_updates % n_drl == 0) and (len(buffer) >= n_srl):
            # if use_ls_dqn and len(buffer) > 1:
                print("performing ls step...")
                if use_dueling_dqn:
                    w_last_before = copy.deepcopy(net.fc2_adv.state_dict())
                else:
                    w_last_before = copy.deepcopy(net.fc2.state_dict())
                batch = buffer.sample(n_srl)
                ls_step(net, tgt_net.target_model, batch, params['gamma'], len(batch), lam=lam,
                        m_batch_size=256, device=device, use_dueling=use_dueling_dqn, use_boosting=use_boosting,
                        use_double_dqn=use_double_dqn)
                # tgt_net.sync()
                if use_dueling_dqn:
                    w_last_after = net.fc2_adv.state_dict()
                else:
                    w_last_after = net.fc2.state_dict()
                weight_diff = torch.sum((w_last_after['weight'] - w_last_before['weight']) ** 2)
                bias_diff = torch.sum((w_last_after['bias'] - w_last_before['bias']) ** 2)
                total_weight_diff = torch.sqrt(weight_diff + bias_diff)
                print(frame_idx, ": total weight difference of ls-update: ", total_weight_diff.item())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            if frame_idx % save_freq == 0:
                if save_for_analysis and frame_idx % n_drl == 0:
                    temp_model_name = model_name + "_" + str(frame_idx)
                    utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                           selector.epsilon, save_replay=True, replay_buffer=buffer.buffer,
                                           name=temp_model_name)
                else:
                    utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                           selector.epsilon, name="-upndown")
