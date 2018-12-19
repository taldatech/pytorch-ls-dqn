#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np
import os

import torch
import basic_model.dqn_model as dqn_model
import basic_model.common as common
import basic_model.ptan.common.wrappers as wrappers
import basic_model.ptan as ptan

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    env = gym.make(args.env)
    env = wrappers.wrap_dqn(env)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # env = wrappers.make_env(DEFAULT_ENV_NAME)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
    # net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    path_to_model_ckpt = './pong_agent_ckpt/pong_agent.pth'
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

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)
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
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

