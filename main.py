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
from utils.parser import get_args, parse_obs_space
from utils.storage import RolloutStorage, NewRolloutStorage
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

    #if not train:
    #    args.num_env_steps = 50000

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

    #envs = gym.make(task)
    #obs_space = env.observation_space
    #act_space = env.action_space
    #action_template = env.action_space.noop()
    env = gym.make(args.task)
    obs_space =env.observation_space
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
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm,
        )
    else:
        raise NotImplementedError

    # storage
    replay_buffer = None
    _, _, non_pixel_shape = parse_obs_space(obs_space)
    add_non_pixel = True if non_pixel_shape > 0 else False
    if args.frame_history_len > 1:
        replay_buffer = ReplayBuffer(100000,
                                    args.frame_history_len, 
                                    non_pixel_shape,
                                    add_non_pixel)

    rollouts = NewRolloutStorage(replay_buffer, args.frame_history_len, 
                            args.num_steps, args.num_processes,
                            obs_space, act_space)

    obs = env.reset()
    #print("reset obs pov size: ",obs['pov'].shape)
    # obs: key: inventory.dirt...
    # (num_processes, size)

    pov, non_pixel_feature = get_obs_features(obs_space, obs)
    #pov, non_pixel_feature = multi_get_obs_features(obs)
    if args.frame_history_len > 1:
        last_stored_frame_idx = replay_buffer.store_frame(pov, non_pixel_feature)
        pov = replay_buffer.encode_recent_observation() / 255.0 # 12 h w
        pov = torch.from_numpy(pov.copy()).reshape(*pov.shape)
    elif args.frame_history_len == 1:
        pov = pov.transpose(2, 0, 1) / 255.0
        pov = torch.from_numpy(pov.copy()).reshape(*pov.shape)
    else:
        raise NotImplementedError

    non_pixel_feature = (torch.tensor(non_pixel_feature) / 180.0)

    #rollouts.obs[0].copy_(pov)
    #rollouts.non_pixel_obs[0].copy_(non_pixel_feature)
    rollouts.temp_obs[0][0].copy_(pov)
    if add_non_pixel:
        rollouts.temp_non_pixel_obs[0][0].copy_(non_pixel_feature)
    rollouts.to(device)

    # ?
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print("Total steps: ", args.num_env_steps)

    ep = 0
    ep_rewards = []
    #mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    #total_rewards = [0 for i in range(args.num_processes)]
    total_rewards = 0

    for j in range(num_updates):
        for process in range(args.num_processes):
            if process > 0:
                rollouts.temp_obs[process][0].copy_(rollouts.temp_obs[process-1][-1])
                if add_non_pixel:
                    rollouts.temp_non_pixel_obs[process][0].copy_(rollouts.temp_non_pixel_obs[process-1][-1])
            
            for step in range(args.num_steps):
                # num_steps = 5
                # Sample actions
                with torch.no_grad():
                    # actor_critic.act output size
                    # actions: torch.Tensor, not list
                    value, actions, action_log_probs = actor_critic.act(rollouts.temp_obs[process][step].unsqueeze(0),
                        rollouts.temp_non_pixel_obs[process][step].unsqueeze(0))

                # value size: batch x 1
                # actions size: torch.Tensor num_processes x num_branches
                # action_log_probs : torch.Tensor num_processes x num_branches
                #print(actions)
                actions_list = actions.squeeze().tolist()

                action = get_actions_continuous(actions_list, act_space, action_template)

                # step:
                #print(actions)
                obs, reward, done, infos = env.step(action)
                #print('.',end='')
                if args.num_env_steps <= 50000:
                    env.render()

                pov, non_pixel_feature = get_obs_features(obs_space, obs)
                #pov, non_pixel_feature = multi_get_obs_features(obs)
                if args.frame_history_len > 1:
                    last_stored_frame_idx = replay_buffer.store_frame(pov, non_pixel_feature)
                    pov = replay_buffer.encode_recent_observation() / 255.0 # 12 h w
                    pov = torch.from_numpy(pov.copy()).reshape(*pov.shape)
                elif args.frame_history_len == 1:
                    pov = pov.transpose(2, 0, 1) / 255.0
                    pov = torch.from_numpy(pov.copy()).reshape(*pov.shape)
                else:
                    raise NotImplementedError

                non_pixel_feature = (torch.tensor(non_pixel_feature) / 180.0)

                total_rewards += reward
                #for i in range(len(reward)):
                #    total_rewards[i] += reward[i]
                reward = torch.tensor([reward]).type(dtype)

                # TODO: may not need bas_masks
                masks = torch.FloatTensor(
                    [0.0] if done else [1.0])
                bad_masks = torch.FloatTensor([1.0])

                if done:
                    ep += 1
                    ep_rewards.append(total_rewards)
                    best_mean_episode_reward = log(j, args.task,ep, np.array(ep_rewards), best_mean_episode_reward)

                    obs = env.reset()
                    pov, non_pixel_feature = get_obs_features(obs_space, obs)
                    #pov, non_pixel_feature = multi_get_obs_features(obs)
                    if args.frame_history_len > 1:
                        last_stored_frame_idx = replay_buffer.store_frame(pov, non_pixel_feature)
                        pov = replay_buffer.encode_recent_observation() / 255.0 # 12 h w
                        pov = torch.from_numpy(pov.copy()).reshape(*pov.shape)
                    elif args.frame_history_len == 1:
                        pov = pov.transpose(2, 0, 1) / 255.0
                        pov = torch.from_numpy(pov.copy()).reshape(*pov.shape)
                    else:
                        raise NotImplementedError
                    non_pixel_feature = (torch.tensor(non_pixel_feature) / 180.0)


                    total_rewards = 0
                # ？
                value = value.squeeze()
                actions = actions.squeeze()
                action_log_probs = action_log_probs.squeeze()
                rollouts.temp_insert(process, step, pov, non_pixel_feature, actions, action_log_probs,
                    value, reward, masks, bad_masks)


        # after step all processes and all steps
        rollouts._transpose()
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

        if j % args.log_interval == 0 and len(ep_rewards) >= 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("----------- Logs -------------")
            if len(ep_rewards) == 0:
                print(
                "Updates {}, num timesteps {}, FPS {} \nThe {}th training episodes,".format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(ep_rewards))
                )
            else:
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
    env.close()


if __name__ == "__main__":
    main()

    
