#!/usr/bin/env python3
import gym
# import argparse
import torch
import torch.optim as optim
import os

from tensorboardX import SummaryWriter

# import dqn_model, common
import ls_dqn_model.dqn_model as dqn_model
import ls_dqn_model.common as common
import ls_dqn_model.ptan as ptan


def save_agent_state(net, optimizer, frame, games, epsilon):
    '''
    This function saves the current state of the DQN (the weights) to a local file.
    '''
    filename = "pong_agent_ls_dqn.pth"
    dir_name = './pong_agent_ckpt'
    full_path = os.path.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'frame_count': frame,
        'games': games,
        'epsilon': epsilon
    }, full_path)
    print("Saved Pong Agent checkpoint @ ", full_path)


def load_agent_state(net, optimizer, selector,  path=None, copy_to_target_network=False, load_optimizer=True):
    """
    This function loads an agent checkpoint.
    Parameters:
        path: path to a checkpoint, e.g `/path/to/dir/ckpt.pth` (str)
        copy_to_target_network: whether or not to copy the loaded training
            DQN parameters to the target DQN, for manual loading (bool)
        load_optimizer: whether or not to restore the optimizer state
    """
    # TODO: add frame_idx, maybe save replay buffer
    if path is None:
        filename = "pong_agent_ls_dqn.pth"
        dir_name = './pong_agent_ckpt'
        full_path = os.path.join(dir_name, filename)
    else:
        full_path = path
    exists = os.path.isfile(full_path)
    if exists:
        if not torch.cuda.is_available():
            checkpoint = torch.load(full_path, map_location='cpu')
        else:
            checkpoint = torch.load(full_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.steps_count = checkpoint['steps_count']
        # self.episodes_seen = checkpoint['episodes_seen']
        selector.epsilon = checkpoint['epsilon']
        # self.num_param_update = checkpoint['num_param_updates']
        print("Checkpoint loaded successfully from ", full_path)
        # # for manual loading a checkpoint
        # if copy_to_target_network:
        #     self.Q_target.load_state_dict(self.Q_train.state_dict())
    else:
        print("No checkpoint found...")


def calc_fqi_matrices(net, tgt_net, batch, gamma, n_srl, m_batch_size=512, device='cpu'):
    """
    :param samples: batch of samples to extract features from (list)
    :param net: network to extract features from (nn.Module)
    :return: A,b parameters for calculating the LS (np.arrays)
    """
    # TODO: implement

    # states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    num_batches = n_srl // m_batch_size
    # curr_batch = 0
    dim = net.fc1.out_features
    num_actions = net.fc2.out_features
    A = torch.zeros([dim * num_actions, dim * num_actions], dtype=torch.float32).to(device)
    b = torch.zeros([dim * num_actions, 1], dtype=torch.float32).to(device)
    for i in range(num_batches):
        idx = i * m_batch_size
        if i == num_batches - 1:
            states, actions, rewards, dones, next_states = common.unpack_batch(batch[idx:])
        else:
            states, actions, rewards, dones, next_states = common.unpack_batch(batch[idx: idx + m_batch_size])
        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        # state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        states_features = net.forward_to_last_hidden(states_v)
        # augmentation
        states_features_aug = torch.zeros([states_features.shape[0], dim * num_actions], dtype=torch.float32).to(device)
        for j in range(states_features.shape[0]):
            position = actions_v[j] * dim
            states_features_aug[j, position:position + dim] = states_features.detach()[j, :]
        # states_features_mat = torch.mm(torch.t(states_features.detach()), states_features.detach())
        states_features_mat = torch.mm(torch.t(states_features_aug), states_features_aug)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        b += torch.mm(torch.t(states_features_aug.detach()), expected_state_action_values.view(-1, 1))
        A += states_features_mat.detach()
    A = (1.0 / n_srl) * A
    b = (1.0 / n_srl) * b
    return A, b


def ls_step(a, b, w, lam=1.0, device='cpu'):
    """
    :param a: A matrix built from features (np.array)
    :param b: b vector built from features and rewards (np.array)
    :param w: weights of the last hidden layer in the DQN (np.array)
    :param lam: regularization parameter for the Least-Square (float)
    :return: w_srl: retrained weights using FQI closed-form solution (np.array)
    """
    # TODO: implement
    num_actions = w.shape[0]
    dim = w.shape[1]
    w = w.view(-1, 1)
    w_srl = torch.mm(torch.inverse(a + lam * torch.eye(num_actions * dim).to(device)), b + lam * w.detach())
    return w_srl.view(num_actions, dim)


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
#    params['epsilon_frames'] = 200000
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
#     args = parser.parse_args()
#     device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    save_freq = 50000
    n_drl = 50000  # steps of DRL between SRL
    n_srl = params['replay_size']  # size of batch in SRL step
    # params['batch_size'] = 256

    writer = SummaryWriter(comment="-" + params['run_name'] + "-ls-dqn")
    net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])  # TODO: change to RMSprop

    load_agent_state(net, optimizer, selector, load_optimizer=False)

    frame_idx = 0
    drl_updates = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon)
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            drl_updates += 1

            # LS-UPDATE STEP
            if (drl_updates % n_drl == 0) and (len(buffer) >= n_srl):
                print("performing ls step...")
                batch = buffer.sample(n_srl)
                a, b = calc_fqi_matrices(net, tgt_net.target_model, batch, params['gamma'],
                                         len(batch), m_batch_size=256, device=device)
                w_last_dict = net.fc2.state_dict()
                w_srl = ls_step(a, b, w_last_dict['weight'], lam=1.0, device=device)
                w_last_dict['weight'] = w_srl
                net.fc2.load_state_dict(w_last_dict)

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            if frame_idx % save_freq == 0:
                save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon)
