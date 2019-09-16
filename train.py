import minerl
import gym

import torch
import torch.optim as optim
import argparse

from model import Branch_DDDQN
from learn import Agent, OptimizerSpec
from DQfN_learn import DQfDAgent
from utils.schedules import *
from utils.replay_buffer import *
from utils.parser import get_args

from builtins import range
from past.utils import old_div

import os
import sys
import time
import json
import random
import logging
import struct
import socket

def main():
    args = get_args()

    # Global Variables
    BATCH_SIZE = 32
    FRAME_HISTORY_LEN = 4
    TARGET_UPDATE_FREQ = 2000
    GAMMA = 0.99
    LEARNING_FREQ = 4
    LEARNING_RATE = 0.00025
    ALPHA = 0.95
    EPS = 0.01
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    RESIZE_WIDTH  = 64
    RESIZE_HEIGHT = 64
    CUDA_VISIBLE_DEVICES = 0

    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS,weight_decay=LAMBDA[3])
    )
    # arguments
    seed = args.seed
    train = False if args.train == 'minitrain' else True
    demo = True if args.demo == 'true' else False
    task = args.task
    PRE_TRAIN_STEPS = args.pre_train_steps
    trajectory_n = 10
    iteration = 3
    convs = [(32,7,3),(64,4,2),(64,3,1)]
    non_pixel_layer = [64]
    in_feature = 7*7*64
    hidden_actions = [128]
    hidden_value = [128]
    aggregator="reduceLocalMean"
    prioritized_replay=True
    prioritized_replay_alpha=0.6
    prioritized_replay_beta0=0.4
    prioritized_replay_beta_iters=5500000
    prioritized_replay_eps=0.0001
    prioritized_demo_replay_eps=0.01
    dueling_dqn = True
    double_dqn = True

    REPLAY_BUFFER_SIZE = args.replay_buffer_size
    EXPLORATION_SCHEDULE = LinearSchedule(args.exploration_schedule, 0.1)
    LEARNING_STARTS = args.learning_starts
    prioritized_replay_beta_iters = args.prioritized_replay_beta_iters

    q_func = Branch_DDDQN

    if not train:
        PRE_TRAIN_STEPS = 1000
        num_reps = 5
        # Global Variables
        BATCH_SIZE = 32
        REPLAY_BUFFER_SIZE = 200000
        FRAME_HISTORY_LEN = 4
        TARGET_UPDATE_FREQ = 1000
        GAMMA = 0.99
        LEARNING_FREQ = 4
        LEARNING_RATE = 0.00025
        ALPHA = 0.95
        EPS = 0.01
        EXPLORATION_SCHEDULE = LinearSchedule(6000, 0.1)
        LEARNING_STARTS = 6000
        prioritized_replay_beta_iters=30000
    else:
        # 6000 steps per ep
        num_reps = 1000
    # logger
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    gpu = True if args.gpu == 'true' else False
    print("current available gpu numbers: %d" %torch.cuda.device_count())
    if (gpu == True):
        if torch.cuda.is_available():
            torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
            print("CUDA Device: %d" %torch.cuda.current_device())

    # Make Environment 
    env = gym.make(task)
    obs_space = env.observation_space
    act_space = env.action_space
    #--------------------------- Agent setting ------------------------------------------------------------
    if not demo:
        agent = Agent(
                env=env,
                q_func=q_func,
                optimizer_spec=optimizer,
                obs_space=obs_space,
                act_space=act_space,
                convs = convs,
                non_pixel_layer=non_pixel_layer,
                in_feature = in_feature,
                hidden_actions = hidden_actions,
                hidden_value = hidden_value,
                aggregator=aggregator,
                exploration=EXPLORATION_SCHEDULE,
                num_episodes=num_reps,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                prioritized_replay=prioritized_replay,
                prioritized_replay_alpha=prioritized_replay_alpha,
                prioritized_replay_beta0=prioritized_replay_beta0,
                prioritized_replay_beta_iters=prioritized_replay_beta_iters,
                prioritized_replay_eps=prioritized_replay_eps,
                batch_size=BATCH_SIZE,gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
        )
        agent.run()
    else:
        agent = DQfDAgent(
                env=env, q_func=q_func, optimizer_spec=optimizer,
                task=task, obs_space=obs_space, act_space=act_space,
                Lambda=LAMBDA,
                convs = convs,
                non_pixel_layer=non_pixel_layer,
                in_feature = in_feature,
                hidden_actions = hidden_actions,
                hidden_value = hidden_value,
                aggregator=aggregator,
                exploration=EXPLORATION_SCHEDULE,
                num_episodes=num_reps,
                pre_train_steps=PRE_TRAIN_STEPS,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                prioritized_replay=prioritized_replay,
                prioritized_replay_alpha=prioritized_replay_alpha,
                prioritized_replay_beta0=prioritized_replay_beta0,
                prioritized_replay_beta_iters=prioritized_replay_beta_iters,
                prioritized_replay_eps=prioritized_replay_eps,
                prioritized_demo_replay_eps=prioritized_demo_replay_eps,
                batch_size=BATCH_SIZE,gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                trajectory_n=trajectory_n,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
        )
        agent.pre_train()
        agent.run()
    #--------------------------- Begin Minecraft game -----------------------------------------------------
    print("-----------------------Training ends-----------------------")


if __name__ == "__main__":
    main()


