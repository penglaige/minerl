import minerl
import gym

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Policy
from ppo import PPO
from utils.schedules import *
from utils.parser import get_args, parse_obs_space, parse_action_space
from utils.storage import RolloutStorage, NewRolloutStorage
from utils.utils import *
from utils.replay_buffer import ReplayBuffer

from builtins import range
from past.utils import old_div
from tqdm import tqdm

import os
import sys
import time
import random

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Demo():
    def __init__(self, task, buffer_size,
                    obs_space,
                    act_space,
                    frame_history_len,
                    add_non_pixel,
                    non_pixel_input_size, 
                    prioritized_replay=False):
        self.task = task
        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.act_space = act_space
        self.add_non_pixel = add_non_pixel
        self.prioritized_replay = prioritized_replay

        if prioritized_replay == True:
            raise NotImplementedError
        else:
            self.buffer = ReplayBuffer(self.buffer_size, frame_history_len,
                    non_pixel_input_size, add_non_pixel)

    def store_demo_data(self):
        """
        Store demo data into the buffer.
        """
        pwd = os.getcwd()
        data_path = pwd + '/human_data'

        data = minerl.data.make(self.task, data_dir=data_path)
        data_dir = data_path + '/' + self.task
        streams = os.listdir(data_dir)

        total_frames = 0
        for stream in streams:
            demo = data.load_data(stream, include_metadata=True)
            try:
                for current_state, action, reward, next_state, done, metadata in demo:
                    if metadata['success'] == False:
                        break
                    total_frames += 1
                    frame, non_pixel_feature = get_obs_features(self.obs_space, current_state)
                    action_spaces, _ = parse_action_space(self.act_space)

                    non_pixel_feature = np.array(non_pixel_feature) / 180.0
                    non_pixel_feature = non_pixel_feature.reshape(1,-1)

                    idx = self.buffer.store_frame(frame, non_pixel_feature, len(action_spaces))

                    act = transfer_actions(action, self.act_space, continuoues=True)
                    
                    reward = np.clip(reward, -1.0, 1.0)
                    done_int = 1 if done else 0
                    self.buffer.store_effect(idx, np.array(act), reward, done_int)

            except RuntimeError:
                print(f"stream {stream} is corrupted!")
                continue

        assert total_frames == self.buffer.num_in_buffer
        print(f"total frames {total_frames}")
        return total_frames

    def sample(self, batch_size):
        if self.add_non_pixel:
            obs_t, act_t, rew_t, obs_tp1, done_mask, non_pixel_obs_t, non_pixel_obs_tp1 = self.buffer.sample(batch_size)
            non_pixel_obs_t = Variable(torch.from_numpy(non_pixel_obs_t)).type(dtype)
            non_pixel_obs_tp1 = Variable(torch.from_numpy(non_pixel_obs_tp1)).type(dtype)
        else:
            obs_t, act_t, rew_t, obs_tp1, done_mask = self.buffer.sample(batch_size)
        
        obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
        act_t = Variable(torch.from_numpy(act_t)).type(dtype)
        rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
        obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
        done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

        if self.add_non_pixel:
            return obs_t, act_t, rew_t, obs_tp1, done_mask, non_pixel_obs_t, non_pixel_obs_tp1
        else:
            return obs_t, act_t, rew_t, obs_tp1, done_mask


    def pre_train(self, actor_critic,optimizer, batch_size, pre_train_nums):
        """
        pre_train is supervised learning
        """
        for _ in tqdm(range(pre_train_nums)):
            non_pixel_obs_t = None
            non_pixel_obs_tp1 = None
            # act_t: (batch, num_branches)
            if self.add_non_pixel:
                obs_t, act_t, rew_t, obs_tp1, done_mask, non_pixel_obs_t, non_pixel_obs_tp1 = self.sample(batch_size)
            else:
                obs_t, act_t, rew_t, obs_tp1, done_mask = self.sample(batch_size)

            # actions: [ batch x k,... ]
            # catagorial, len of action numbers, log loss
            # gaussian: mean, mse loss?
            action_probs = actor_critic.demo_act(obs_t, non_pixel_obs_t)

            for i in range(len(action_probs)):
                action_prob = action_probs[i].reshape(batch_size, -1)
                action_label = act_t[:,i].reshape(batch_size,-1)
                #print("action_prob size: ", action_prob.size())
                #print(action_prob)
                #print("action_label size: ", action_label.size())
                #print(action_label)

                if action_prob.size(1) == 1:
                    # MSE loss
                    mseloss = nn.MSELoss()
                    branch_loss = mseloss(action_prob, action_label)
                else:
                    # negative log likelihood loss
                    celoss = nn.CrossEntropyLoss()
                    action_label = action_label.reshape(batch_size).type(dlongtype)
                    branch_loss = celoss(action_prob, action_label)

                if i == 0:
                    loss = branch_loss
                else:
                    loss += branch_loss
            
            loss /= len(action_probs)
            
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            






