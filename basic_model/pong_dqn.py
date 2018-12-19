#!/usr/bin/env python3
import gym
# import argparse
import torch
import torch.optim as optim
import os

from tensorboardX import SummaryWriter

# import dqn_model, common
import basic_model.dqn_model as dqn_model
import basic_model.common as common
import basic_model.ptan as ptan


def save_agent_state(net, optimizer, frame, games, epsilon):
    '''
    This function saves the current state of the DQN (the weights) to a local file.
    '''
    filename = "pong_agent.pth"
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
    '''
    This function loads an agent checkpoint.
    Parameters:
        path: path to a checkpoint, e.g `/path/to/dir/ckpt.pth` (str)
        copy_to_target_network: whether or not to copy the loaded training
            DQN parameters to the target DQN, for manual loading (bool)
        load_optimizer: whether or not to restore the optimizer state
    '''
    if path is None:
        filename = "pong_agent.pth"
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

    writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
    net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device) # need to divide FC layers
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate']) # change to RMSprop

    load_agent_state(net, optimizer, selector, load_optimizer=False)

    frame_idx = 0

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
            # LS-UPDATE STEP

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            if frame_idx % save_freq == 0:
                save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards), selector.epsilon)
