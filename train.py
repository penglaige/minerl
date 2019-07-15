import minerl
import gym

import torch
import torch.optim as optim
import argparse

from model import Branches_dueling_DQN
from learn import Agent, OptimizerSpec
from utils.schedules import *

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

# Global Variables
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 500000
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 2000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1200000, 0.1)
LEARNING_STARTS = 60000

RESIZE_WIDTH  = 64
RESIZE_HEIGHT = 64
CUDA_VISIBLE_DEVICES = 0

optimizer = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
)

# argparse
train = True

if not train:
    num_reps = 5
    # Global Variables
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 50000
    FRAME_HISTORY_LEN = 4
    TARGET_UPDATE_FREQ = 2000
    GAMMA = 0.99
    LEARNING_FREQ = 4
    LEARNING_RATE = 0.00025
    ALPHA = 0.95
    EPS = 0.01
    EXPLORATION_SCHEDULE = LinearSchedule(6000, 0.1)
    LEARNING_STARTS = 6000
else:
    num_reps = 300
# logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Command Parser
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
gpu = None
print("current available gpu numbers: %d" %torch.cuda.device_count())
if (gpu != None):
    if torch.cuda.is_available():
        torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
        print("CUDA Device: %d" %torch.cuda.current_device())

seed = 0

# models parameter
action_space = []
convs = [(32,7,3),(64,4,2),(64,3,1)]
non_pixel_layer = [64]
non_pixel_input_size = 2
in_feature = 7*7*64
hidden_actions = [128]
hidden_value = [128]
num_actions_for_branch = [2,2,2,2,2,2,2,2,2,36,36]
aggregator="reduceLocalMean"
prioritized_replay=True
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=2e6
dueling_dqn = True
double_dqn = True

# Environment 
env = gym.make('MineRLNavigateDense-v0')
action = env.action_space.noop()
for key in action:
    if(key != 'camera'):
        action_space.append(key)
action_space.append('camera')
num_branches = len(action_space) + 1

#--------------------------- Agent setting ------------------------------------------------------------
agent = Agent(
        env=env,
        q_func=Branches_dueling_DQN,
        optimizer_spec=optimizer,
        action_space=action_space,
        convs = convs,
        non_pixel_layer=non_pixel_layer,
        in_feature = in_feature,
        non_pixel_input_size = non_pixel_input_size,
        hidden_actions = hidden_actions,
        hidden_value = hidden_value,
        num_branches = num_branches,
        num_actions_for_branch = num_actions_for_branch,
        aggregator=aggregator,
        exploration=EXPLORATION_SCHEDULE,
        num_episodes=num_reps,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        prioritized_replay=prioritized_replay,
        prioritized_replay_alpha=prioritized_replay_alpha,
        prioritized_replay_beta0=prioritized_replay_beta0,
        prioritized_replay_beta_iters=prioritized_replay_beta_iters,
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

#--------------------------- Begin Minecraft game -----------------------------------------------------
agent.run()
#--------------------------- Begin Minecraft game -----------------------------------------------------
print("-----------------------Training ends-----------------------")



