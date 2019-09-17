# reference: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# for ppo
import minerl
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Policy
from ppo import PPO
from utils.schedules import *
from utils.parser import get_args
from utils.storage import RolloutStorage
from utils.utils import *
from utils.replay_buffer import ReplayBuffer

from builtins import range
from past.utils import old_div

import os
import sys
import time
import copy
import glob
import json
import random
import logging
import struct
import socket
from collections import deque

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.gpu and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # arguments
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    CUDA_VISIBLE_DEVICES = 0
    seed = args.seed
    train = args.train
    demo = args.demo
    task = args.task
    iteration = 3
    convs = [(32,7,3),(64,4,2),(64,3,1)]
    non_pixel_layer = [64]
    in_feature = 7*7*64
    hidden_actions = [128]
    hidden_value = [128]
    aggregator = "reduceLocalMean"
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if not train:
        args.num_env_steps = 50000

    base_kwargs = {
        'non_pixel_layer':non_pixel_layer,
        'convs': convs,
        'frame_history_len':args.frame_history_len,
        'in_feature':in_feature,
        'hidden_actions':hidden_actions,
        'hidden_value':hidden_value,
        'aggregator':aggregator
    }

    # logger
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # threads and device
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.gpu else "cpu")
    print("device:", device)
    gpu = args.gpu
    if (gpu == True):
        print("current available gpu numbers: %d" %torch.cuda.device_count())
        if torch.cuda.is_available():
            torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
            print("CUDA Device: %d" %torch.cuda.current_device())

    # envs
    env = gym.make(task)
    obs_space = env.observation_space
    act_space = env.action_space
    action_template = env.action_space.noop()

    # policy
    actor_critic = Policy(obs_space, 
                        act_space,
                        base_kwargs=base_kwargs)
    actor_critic.to(device)

    # algorithm
    if args.algo == 'ppo':
        agent = PPO(actor_critic,
                    args.clip_param,
                    args.ppo_epoch,
                    args.num_mini_batch,
                    args.value_loss_coef,
                    args.entropy_coef,
                    lr=7e-4,
                    eps=1e-5,
                    max_grad_norm=args.max_grad_norm,
        )
    else:
        raise NotImplementedError

    # storage
    replay_buffer = None
    rollouts = RolloutStorage(replay_buffer, args.frame_history_len, 
                            args.num_steps, args.num_processes,
                            obs_space, act_space)

    obs = env.reset()

    pov, non_pixel_feature = get_obs_features(obs_space, obs)
    pov = pov.transpose(2, 0, 1) / 250.0
    # TODO: replace 1 with num_process
    pov = torch.from_numpy(pov.copy()).reshape(1,*pov.shape)
    non_pixel_feature = (torch.tensor(non_pixel_feature) / 180.0).reshape(1,-1)

    rollouts.obs[0].copy_(pov)
    rollouts.non_pixel_obs[0].copy_(non_pixel_feature)
    rollouts.to(device)

    # ?
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print("Total steps: ", args.num_env_steps)

    ep = 0
    ep_rewards = []
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    total_rewards = 0

    for j in range(num_updates):
        
        for step in range(args.num_steps):
            # num_steps = 5
            # Sample actions
            with torch.no_grad():
                # actor_critic.act output size
                # actions: torch.Tensor, not list
                value, actions, action_log_probs = actor_critic.act(rollouts.obs[step],
                    rollouts.non_pixel_obs[step])

            # value size: batch x 1
            # actions size: torch.Tensor batch x num_branches
            # action_log_probs : torch.Tensor batch x num_branches
            actions_list = actions.squeeze().tolist()

            action = get_actions_continuous(actions_list, act_space, action_template)

            # step:
            #print(action)
            obs, reward, done, info = env.step(action)
            if not train:
                env.render()

            pov, non_pixel_feature = get_obs_features(obs_space, obs)
            pov = pov.transpose(2, 0, 1) / 250.0
            pov = (torch.from_numpy(pov.copy())).reshape(1,*pov.shape)
            non_pixel_feature = (torch.tensor(non_pixel_feature) / 180.0).reshape(1,-1).type(dtype)
            # TODO:replace by num process
            total_rewards += reward
            reward = torch.tensor([reward]).reshape(1,-1).type(dtype)

            # If done then clean the history of observations.
            # implement for 1 process
            # TODO: may not need bas_masks
            masks = torch.FloatTensor(
                [[0.0] if done else [1.0]])
            bad_masks = torch.FloatTensor([[1.0]])

            rollouts.insert(pov, non_pixel_feature, actions, action_log_probs,
                value, reward, masks, bad_masks)

            # If done:
            if done:
                ep += 1
                ep_rewards.append(total_rewards)
                log(j, ep, np.array(ep_rewards), mean_episode_reward, best_mean_episode_reward)

                total_rewards = 0

                # reset
                obs = env.reset()

                pov, non_pixel_feature = get_obs_features(obs_space, obs)
                pov = pov.transpose(2, 0, 1) / 250.0
                # TODO: replace 1 with num_process
                pov = torch.from_numpy(pov.copy()).reshape(1,*pov.shape)
                non_pixel_feature = (torch.tensor(non_pixel_feature) / 180.0).reshape(1,-1)
                
                # TODO: how to deal with reset
                terminal_actions = torch.zeros(actions.size())
                terminal_action_log_probs = torch.zeros(action_log_probs.size())
                terminal_value = torch.zeros(value.size())
                terminal_reward = torch.zeros(reward.size())
                rollouts.insert(pov, non_pixel_feature, terminal_actions, terminal_action_log_probs, terminal_value,
                    terminal_reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.non_pixel_obs[-1])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
            args.gae_lambda, args.use_proper_time_limits)

        # TODO: minibathc = 32, 1 processes x 10 step should larger than 32
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_model_dir != '':
            save_path = os.path.join(args.save_model_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(actor_critic, os.path.join(save_path, args.task + ".pt"))

        if j % args.log_interval == 0 and len(ep_rewards) > 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("----------- Logs -------------")
            print(
                "Updates {}, num timesteps {}, FPS {} \nThe {}th training episodes,\nmean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(ep_rewards),np.mean(ep_rewards),
                    np.median(ep_rewards), np.min(ep_rewards),
                    np.max(ep_rewards)
                )

            )
    
    print("-----------------------Training ends-----------------------")



if __name__ == "__main__":
    main()

    