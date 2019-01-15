import gym
import argparse
import torch
import torch.optim as optim
import collections

from tensorboardX import SummaryWriter

import ls_dqn_model.utils.dqn_model as dqn_model
from ls_dqn_model.utils.hyperparameters import HYPERPARAMS
from ls_dqn_model.utils.agent import DQNAgent, TargetNet
from ls_dqn_model.utils.actions import EpsilonGreedyActionSelector, ArgmaxActionSelector
import ls_dqn_model.utils.experience as experience
import ls_dqn_model.utils.utils as utils
from ls_dqn_model.utils.srl_algorithms import ls_step, ls_step_dueling
import ls_dqn_model.utils.wrappers as wrappers
import numpy as np
import random
import time
import os
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and play an LS-DQN agent")
    # modes
    parser.add_argument("-t", "--train", help="train or continue training an agent",
                        action="store_true")
    parser.add_argument("-h", "--lsdqn", help="use LS-DQN",
                        action="store_true")
    parser.add_argument("-j", "--boosting", help="use boosting",
                        action="store_true")
    parser.add_argument("-u", "--double", help="use double dqn",
                        action="store_true")
    parser.add_argument("-f", "--dueling", help="use dueling dqn",
                        action="store_true")
    parser.add_argument("-p", "--play", help="play the environment using an a pretrained agent",
                        action="store_true")
    parser.add_argument("-y", "--path", type=str, help="path to agent checkpoint, for playing")
    # arguments
    # for training and playing
    parser.add_argument("-n", "--name", type=str,
                        help="model name, for saving and loading,"
                             " if not set, training will continue from a pretrained checkpoint")
    parser.add_argument("-e", "--env", type=int,
                        help="environment to play: pong, boxing, breakout, breakout-small, invaders")
    # for training
    parser.add_argument("-d", "--decay_rate", type=int,
                        help="number of episodes for epsilon decaying, default: 100000")
    parser.add_argument("-o", "--optimizer", type=str,
                        help="optimizing algorithm ('RMSprop', 'Adam'), deafult: 'Adam'")
    parser.add_argument("-r", "--learn_rate", type=float,
                        help="learning rate for the optimizer, default: 0.0001")
    parser.add_argument("-l", "--lam", type=float,
                        help="regularization parameter value, default: 1, 10000 (boosting)")
    parser.add_argument("-g", "--gamma", type=float,
                        help="gamma parameter for the Q-Learning, default: 0.99")
    parser.add_argument("-s", "--buffer_size", type=int,
                        help="Replay Buffer size, default: 1000000")
    parser.add_argument("-a", "--n_drl", type=int,
                        help="number of drl updates before an srl update, default: 500000")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="number of samples in each batch, default: 32")
    parser.add_argument("-i", "--steps_to_start_learn", type=int,
                        help="number of steps before the agents starts learning, default: 10000")
    parser.add_argument("-c", "--target_update_freq", type=int,
                        help="number of steps between copying the weights to the target DQN, default: 10000")
    parser.add_argument("-x", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")

    args = parser.parse_args()

    if not args.env or HYPERPARAMS.get(args.env) is None:
        raise SystemExit("No valid environment")
    else:
        params = HYPERPARAMS[args.env]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(params['env_name'])
    env = wrappers.wrap_dqn(env)

    if args.lsdqn:
        use_ls_dqn = True
    else:
        use_ls_dqn = False
    if args.boosting:
        use_boosting = True
        lam = 10000
    else:
        use_boosting = False
        lam = 1
    if args.double:
        use_double_dqn = True
    else:
        use_double_dqn = False
    if args.dueling:
        use_dueling_dqn = True
    else:
        use_dueling_dqn = False
    # Training
    if args.train:
        if args.name:
            model_name = args.name
        else:
            model_name = ''
        if args.decay_rate:
            params['epsilon_frames'] = args.decay_rate
        if args.learn_rate:
            params['learning_rate'] = args.learn_rate
        if args.lam:
            lam = args.lam
        if args.gamma:
            params['gamma'] = args.gamma
        if args.buffer_size:
            params['replay_size'] = args.buffer_size
        if args.n_drl:
            n_drl = args.n_drl
        else:
            n_drl = 500000  # steps of DRL between SRL
        if args.batch_size:
            params['batch_size'] = args.batch_size
        if args.steps_to_start_learn:
            params['replay_initial'] = args.steps_to_start_learn
        if args.target_update_freq:
            params['target_net_sync'] = args.target_update_freq

        # training_random_seed = 10
        save_freq = 50000
        n_srl = params['replay_size']  # size of batch in SRL step
        # num_srl_updates = 3  # number of to SRL updates to perform

        # use_constant_seed = False  # to compare performance independently of the randomness
        # save_for_analysis = False  # save also the replay buffer for later analysis

        if use_ls_dqn:
            print("using ls-dqn with lambda:", str(lam))
            model_name += "-LSDQN-LAM-" + str(lam) + "-" + str(int(1.0 * n_drl / 1000)) + "K"
        else:
            model_name += "-DQN"
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

        model_saving_path = './agent_ckpt/agent_' + model_name + ".pth"
        # if use_constant_seed:
        #     model_name += "-SEED-" + str(training_random_seed)
        #     np.random.seed(training_random_seed)
        #     random.seed(training_random_seed)
        #     env.seed(training_random_seed)
        #     torch.manual_seed(training_random_seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(training_random_seed)
        #     print("training using constant seed of ", training_random_seed)

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
        if args.optimizer and args.optimizer == 'RMSprop':
            optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
        else:
            optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'])

        utils.load_agent_state(net, optimizer, selector, load_optimizer=False, env_name=params['env_name'],
                               path=model_saving_path)

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
                        # if save_for_analysis:
                        #     temp_model_name = model_name + "_" + str(frame_idx)
                        #     utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                        #                            selector.epsilon, save_replay=True,
                        #                            replay_buffer=buffer.buffer,
                        #                            name=temp_model_name)
                        # else:
                        #     utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                        #                            selector.epsilon, name='-boxing')
                        utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                               selector.epsilon, path=model_saving_path)
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
                    print("performing ls step...")
                    batch = buffer.sample(n_srl)
                    if use_dueling_dqn:
                        ls_step_dueling(net, tgt_net.target_model, batch, params['gamma'], len(batch), lam=lam, m_batch_size=256,
                                        device=device,
                                        use_boosting=use_boosting, use_double_dqn=use_double_dqn)
                    else:
                        ls_step(net, tgt_net.target_model, batch, params['gamma'], len(batch), lam=lam,
                                m_batch_size=256, device=device, use_boosting=use_boosting,
                                use_double_dqn=use_double_dqn)

                if frame_idx % params['target_net_sync'] == 0:
                    tgt_net.sync()
                utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                       selector.epsilon, path=model_saving_path)
                # if frame_idx % save_freq == 0:
                #     if save_for_analysis and frame_idx % n_drl == 0:
                #         temp_model_name = model_name + "_" + str(frame_idx)
                #         utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                #                                selector.epsilon, save_replay=True, replay_buffer=buffer.buffer,
                #                                name=temp_model_name)
                #     else:
                #         utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                #                                selector.epsilon, path=model_saving_path)

    elif args.play:
        # play
        if args.path:
            path_to_model_ckpt = args.path
        else:
            raise SystemExit("must include path to agent checkpoint")
        FPS = 25
        if args.record:
            env = gym.wrappers.Monitor(env, args.record)
        if use_dueling_dqn:
            net = dqn_model.DuelingLSDQN(env.observation_space.shape, env.action_space.n).to(device)
        else:
            net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
        # net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        path_to_model_ckpt = './agent_ckpt/agent_ls_dqn_-boxing.pth'
        exists = os.path.isfile(path_to_model_ckpt)
        if exists:
            if not torch.cuda.is_available():
                checkpoint = torch.load(path_to_model_ckpt, map_location='cpu')
            else:
                checkpoint = torch.load(path_to_model_ckpt)
            net.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded checkpoint from ", path_to_model_ckpt)
        else:
            raise SystemExit("Checkpoint File Not Found")

        selector = ArgmaxActionSelector()
        agent = DQNAgent(net, selector, device=device)
        state = env.reset()
        total_reward = 0.0
        c = collections.Counter()

        while True:
            start_ts = time.time()
            if args.visualize:
                env.render()
            # state_v = torch.tensor(np.array([state], copy=False))
            # state_v = ptan.agent.default_states_preprocessor(state)
            # q_vals = net(state_v).data.numpy()[0]
            # action = np.argmax(q_vals)
            action, _ = agent([state])
            # print(action)
            c[action[0]] += 1
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                env.close()
                break
            if args.visualize:
                delta = 1 / FPS - (time.time() - start_ts)
                if delta > 0:
                    time.sleep(delta)
        print("Total reward: %.2f" % total_reward)
        print("Action counts:", c)
        if args.record:
            env.env.close()
    else:
        raise SystemExit("must choose between train or play")
