"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
# todo
"""
import minerl
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from utils.replay_buffer import *
from utils.schedules import *
from logger import Logger
import time
import pickle
from tqdm import tqdm

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

#CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def to_np(x):
    return x.data.cpu().numpy()

# learner
class DQfDAgent():
    """
    A branch_duling_dqn learner agent.
    """

    def __init__(self,env,q_func,optimizer_spec,
        task,
        action_space,
        convs = [(32,7,3),(64,4,2),(64,3,1)],
        non_pixel_layer=[64],
        non_pixel_input_size = 2,
        add_non_pixel=False,
        in_feature = 7*7*64,
        hidden_actions = [128],
        hidden_value = [128],
        num_branches = 11,
        num_actions_for_branch = [2,2,2,2,2,2,2,2,2,36,36],
        aggregator="reduceLocalMean",
        exploration=LinearSchedule(200000, 0.1),
        stopping_criterion=None,
        num_episodes=1000,
        #TODO
        pre_train_steps=100000,
        replay_buffer_size=1000000,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=1600000,
        prioritized_replay_eps=0.00001,
        prioritized_demo_replay_eps= 1.0,
        batch_size=32,gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        img_h=64,
        img_w=64,
        img_c=3,
        target_update_freq=10000,
        double_dqn=False,
        dueling_dqn=False
        ):
        """Run Deep Q-learning algorithm.
        You can specify your own convnet using q_func.
        All schedules are w.r.t. total number of steps taken in the environment.
        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        env_id: string
            gym environment id for model saving.
        q_func: function
            Model to use for computing the q function.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        exploration: rl_algs.deepq.utils.schedules.Schedule
            schedule for probability of chosing random action.
        stopping_criterion: (env, t) -> bool
            should return true when it's ok for the RL algorithm to stop.
            takes in env and the number of steps executed so far.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to max_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the unified TD error for updating priorities.
            Erratum: The camera-ready copy of this paper incorrectly reported 1e-8. 
            The value used to produece the results is 1e8.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        """
        
        ###############
        # BUILD MODEL #
        ###############
        self.logger = Logger('./logs')
        self.env = env
        self.q_func = q_func
        self.task = task
        self.convs = convs
        self.non_pixel_layer = non_pixel_layer
        self.non_pixel_input_size = non_pixel_input_size
        self.add_non_pixel = add_non_pixel
        self.in_feature = in_feature
        self.hidden_actions = hidden_actions
        self.hidden_value = hidden_value
        self.num_branches = num_branches
        self.num_actions_for_branch = num_actions_for_branch
        self.action_space = action_space
        self.aggregator = aggregator
        self.optimizer_spec = optimizer_spec
        self.exploration = exploration
        self.num_episodes = num_episodes
        self.pre_train_steps = pre_train_steps
        self.stopping_criterion = stopping_criterion
        self.replay_buffer_size = replay_buffer_size
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_demo_replay_eps = prioritized_demo_replay_eps
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.frame_history_len = frame_history_len
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.input_shape = (self.img_h, self.img_w, self.frame_history_len * self.img_c)
        self.in_channels = self.input_shape[2]

        # define Q target and Q
        if not self.add_non_pixel:
            self.Q = self.q_func(self.convs, self.in_channels, 
                                self.in_feature, self.hidden_actions, 
                                self.hidden_value, self.num_branches, 
                                self.num_actions_for_branch, self.aggregator).type(dtype)
            self.Q_target = self.q_func(self.convs, self.in_channels, 
                                self.in_feature, self.hidden_actions, 
                                self.hidden_value, self.num_branches, 
                                self.num_actions_for_branch, self.aggregator).type(dtype)
        else:
            # print("with non pixel feature")
            self.Q = self.q_func(self.non_pixel_input_size, self.non_pixel_layer,
                                self.convs, self.in_channels, 
                                self.in_feature, self.hidden_actions, 
                                self.hidden_value, self.num_branches, 
                                self.num_actions_for_branch, self.aggregator).type(dtype)
            self.Q_target = self.q_func(self.non_pixel_input_size, self.non_pixel_layer,
                                self.convs, self.in_channels, 
                                self.in_feature, self.hidden_actions, 
                                self.hidden_value, self.num_branches, 
                                self.num_actions_for_branch, self.aggregator).type(dtype)

        # initialize optimizer
        self.optimizer = self.optimizer_spec.constructor(self.Q.parameters(), **self.optimizer_spec.kwargs)

        # create repaly buffer 
        if(prioritized_replay == True):
            self.replay_buffer = demoReplayBuffer(self.replay_buffer_size, self.frame_history_len, 
                                                        self.prioritized_replay_alpha, self.num_branches,
                                                        self.non_pixel_input_size,self.add_non_pixel)
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                            initial_p=prioritized_replay_beta0,
                                            final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len,self.non_pixel_input_size, self.add_non_pixel)
            self.beta_schedule = None
        self.demo_size = self.store_demo_data()
        self.replay_buffer.demo_size = self.demo_size

        ###### RUN SETTING ####
        self.t = 0
        self.num_param_updates = 0
        self.mean_episode_reward      = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = None
        
        self.Log_Every_N_Steps = 600000
        self.Save_Model_Every_N_Steps = 600000
        self.Save_Model_Every_N_EPs = 50
        self.Save_Reward_Every_N_EPs = 10
        
        ##########################
        self.separate = [self.num_actions_for_branch[0]]
        for i in range(1,len(self.num_actions_for_branch)-1):
            self.separate.append(self.separate[-1] + self.num_actions_for_branch[i])
        self.separate = np.array(self.separate)

    def run(self):
        ep_rewards = []
        for ep in range(1, self.num_episodes + 1):
            total_rewards = 0
            self.last_obs = self.env.reset()

            done = False
            while not done:
                ### Step the agent and store the transition
                # store last frame, returned idx used later
                last_pov = self.last_obs["pov"]  # pixel feature
                last_compass = self.last_obs["compassAngle"] # non_pixel_feature
                last_dirt = self.last_obs["inventory"]["dirt"]  # non_pixel_feature
                last_non_pixel_feature = np.array([last_compass / 180.0, last_dirt / 64.0]).reshape(1,2)
                last_stored_frame_idx = self.replay_buffer.store_frame(last_pov, last_non_pixel_feature)

                # get observations to input to Q netword
                observations = self.replay_buffer.encode_recent_observation()
                if self.add_non_pixel:
                    non_pixel_feature = self.replay_buffer.encode_recent_non_pixel_feature()

                if self.t < self.learning_starts:
                    act = self.get_random_action()
                    action = self.get_action(act)
                else:
                    # epsilon greedy exploration
                    sample = random.random()
                    threshold = self.exploration.value(self.t)
                    if sample > threshold:
                        obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
                        non_pixel_feature = torch.from_numpy(non_pixel_feature).type(dtype)
                        with torch.no_grad():
                            if self.add_non_pixel:
                                action = self.Q(obs,non_pixel_feature)
                            else:
                                action = self.Q(obs)
                            
                            action = self.divide(action, self.separate)

                            act = []
                            for a in action:
                                #print("a: ",a)
                                act.append(torch.argmax(a).item())
                            action = self.get_action(act)
                            #print("action: ",action)
                    else:
                        # act: []
                        act = self.get_random_action()
                        action = self.get_action(act)

                obs, reward, done, info = self.env.step(action)
                self.env.render()
                total_rewards += reward

                # clipping the reward, noted in nature paper
                reward = np.clip(reward, -1.0, 1.0)

                # store effect of action
                if done:
                    done_int = 1
                else:
                    done_int = 0
                self.replay_buffer.store_effect(last_stored_frame_idx, np.array(act), reward, done_int)
                self.t += 1

                if done:
                    print(f"----------Episode {ep} end!")
                    ep_rewards.append(total_rewards)
                    self.log(ep, ep_rewards)
                
                # uodate last obs
                self.last_obs = obs

                # Perform experience replay and train the network
                # if the replay buffer contains enough samples
                if (self.t >= self.learning_starts + self.demo_size and self.t % self.learning_freq == 0 and self.replay_buffer.can_sample(self.batch_size)):
                    td_error, idxes = self.train()
                    if self.prioritized_replay:
                        self.update_priorities(td_error, self.prioritized_replay_eps,idxes)

                # Save model
                if self.Save_Model_Every_N_EPs > 0:
                    if ep % self.Save_Model_Every_N_EPs == 0:
                        if not os.path.exists("models"):
                            os.makedirs("models")
                        model_save_path = f"models/ep{self.t}.model" 
                        torch.save(self.Q.state_dict(), model_save_path)
                else:
                    if self.t % self.Save_Model_Every_N_Steps == 0:
                        if not os.path.exists("models"):
                            os.makedirs("models")
                        model_save_path = f"models/t{self.t}.model" 
                        torch.save(self.Q.state_dict(), model_save_path)

    def train(self, pre_train=False):
        # Sample transition batch from replay memory
        # done_mash = 1 if next state is end of episode
        if self.add_non_pixel:
            obs_t, act_t, rew_t, obs_tp1, done_mask, non_pixel_obs_t, non_pixel_obs_tp1, weights, idxes = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.t))
            non_pixel_obs_t = Variable(torch.from_numpy(non_pixel_obs_t)).type(dtype)
            non_pixel_obs_tp1 = Variable(torch.from_numpy(non_pixel_obs_tp1)).type(dtype)
        else:
            obs_t, act_t, rew_t, obs_tp1, done_mask, weights, idxes = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.t))
        obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
        act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
        rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
        obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
        done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

        if self.add_non_pixel:
            # input batches to networks
            # get the Q values for current observations for each action dimension
            q_values = self.Q(obs_t,non_pixel_obs_t)
            q_values = self.divide(q_values, self.separate)
            #q_values = np.hsplit(q_values, self.separate)
            q_s_a = []
            for dim in range(self.num_branches):
                selected_a = act_t[:,dim].view(act_t.size(0),-1)
                q_value = q_values[dim]
                q_s_a_dim = q_value.gather(1, selected_a)
                q_s_a.append(q_s_a_dim)
                #if dim == 0:
                #    q_s_a = q_s_a_dim
                #else:
                #    q_s_a = torch.cat((q_s_a, q_s_a_dim),1)

            # calculate target Q value:
            if self.double_dqn:
                # ---------------
                #   double DQN
                # ---------------

                # get the Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s', a', theta_i)) wrt a'
                q_tp1_values = self.Q(obs_tp1,non_pixel_obs_tp1).detach()
                q_tp1_values = self.divide(q_tp1_values, self.separate)
                a_prime = []
                for dim in range(self.num_branches):
                    q_tp1_value = q_tp1_values[dim]
                    _, a_prime_dim = q_tp1_value.max(1)
                    a_prime.append(a_prime_dim)

                # get Q values from frozen network for next state and chosen action
                q_target_tp1_values = self.Q_target(obs_tp1,non_pixel_obs_tp1).detach()
                q_target_tp1_values = self.divide(q_target_tp1_values, self.separate)

                expected_state_action_values = []
                
                for dim in range(self.num_branches):
                    q_target_tp1_value = q_target_tp1_values[dim]
                    q_target_s_a_prime_dim = q_target_tp1_value.gather(1,a_prime[dim].unsqueeze(1))
                    q_target_s_a_prime_dim = q_target_s_a_prime_dim.squeeze()

                    #if current state is end of episode, then there if no next Q value
                    q_target_s_a_prime_dim = (1 - done_mask) * q_target_s_a_prime_dim

                    # TODO ： mean td error
                    expected_state_action_values_dim = (rew_t + self.gamma * q_target_s_a_prime_dim).view(self.batch_size, -1)
                    #print("expected_state_action_values_dim size: ",expected_state_action_values_dim.size())
                    expected_state_action_values.append(expected_state_action_values_dim)

                # calculate loss
                branch_losses = []
                for dim in range(self.num_branches):
                    td_error_dim = q_s_a[dim] - expected_state_action_values[dim]
                    if dim == 0:
                        td_error = torch.abs(td_error_dim)
                    else:
                        td_error += torch.abs(td_error_dim)
                    loss_dim = F.smooth_l1_loss(q_s_a[dim], expected_state_action_values[dim])
                    #print("loss_dim: ",loss_dim)
                    branch_losses.append(loss_dim)
                
                loss = sum(branch_losses) / self.num_branches

            else:
                # -----------------
                #   regular DQN
                # -----------------

                # get the Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = self.Q_target(obs_tp1,non_pixel_obs_tp1).detach()
                q_tp1_values = self.divide(q_tp1_values, self.separate)
                q_s_a_prime = []
                a_prime = []
                loss = 0
                for dim in range(self.num_branches):
                    q_tp1_value = q_tp1_values[dim]
                    q_s_a_prime_dim, a_prime_dim = q_tp1_value.max(1)

                    q_s_a_prime_dim = (1 - done_mask) * q_s_a_prime_dim
                    q_s_a_prime.append(q_s_a_prime_dim)
                    a_prime.append(a_prime_dim)

                    expected_state_action_values_dim = rew_t + self.gamma * q_s_a_prime_dim

                    td_error_dim = q_s_a[dim] - expected_state_action_values_dim
                    if dim == 0:
                        td_error = torch.abs(td_error_dim)
                    else:
                        td_error += torch.abs(td_error_dim)

                    loss_dim = F.smooth_l1_loss(q_s_a[dim],expected_state_action_values_dim)
                    loss += loss_dim
                loss /= self.num_branches
            # clip the error and flip--old
            #clipped_error = -1.0 * error.clamp(-1, 1)

        else:
            # input batches to networks
            # get the Q values for current observations for each action dimension
            q_values = self.Q(obs_t)
            q_values = self.divide(q_values, self.separate)
            #q_values = np.hsplit(q_values, self.separate)
            q_s_a = []
            for dim in range(self.num_branches):
                selected_a = act_t[:,dim].view(act_t.size(0),-1)
                q_value = q_values[dim]
                q_s_a_dim = q_value.gather(1, selected_a)
                q_s_a.append(q_s_a_dim)
                #if dim == 0:
                #    q_s_a = q_s_a_dim
                #else:
                #    q_s_a = torch.cat((q_s_a, q_s_a_dim),1)

            # calculate target Q value:
            if self.double_dqn:
                # ---------------
                #   double DQN
                # ---------------

                # get the Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s', a', theta_i)) wrt a'
                q_tp1_values = self.Q(obs_tp1).detach()
                q_tp1_values = self.divide(q_tp1_values, self.separate)
                a_prime = []
                for dim in range(self.num_branches):
                    q_tp1_value = q_tp1_values[dim]
                    _, a_prime_dim = q_tp1_value.max(1)
                    a_prime.append(a_prime_dim)

                # get Q values from frozen network for next state and chosen action
                q_target_tp1_values = self.Q_target(obs_tp1).detach()
                q_target_tp1_values = self.divide(q_target_tp1_values, self.separate)

                expected_state_action_values = []
                
                for dim in range(self.num_branches):
                    q_target_tp1_value = q_target_tp1_values[dim]
                    q_target_s_a_prime_dim = q_target_tp1_value.gather(1,a_prime[dim].unsqueeze(1))
                    q_target_s_a_prime_dim = q_target_s_a_prime_dim.squeeze()

                    #if current state is end of episode, then there if no next Q value
                    q_target_s_a_prime_dim = (1 - done_mask) * q_target_s_a_prime_dim

                    # TODO ： mean td error
                    expected_state_action_values_dim = (rew_t + self.gamma * q_target_s_a_prime_dim).view(self.batch_size, -1)
                    #print("expected_state_action_values_dim size: ",expected_state_action_values_dim.size())
                    expected_state_action_values.append(expected_state_action_values_dim)

                # calculate loss
                branch_losses = []
                for dim in range(self.num_branches):
                    td_error_dim = q_s_a[dim] - expected_state_action_values[dim]
                    if dim == 0:
                        td_error = torch.abs(td_error_dim)
                    else:
                        #td_error = torch.cat((td_error, torch.abs(td_error_dim)),1)
                        td_error += td_error_dim
                    #td_error /= self.num_brunches

                    loss_dim = F.smooth_l1_loss(q_s_a[dim], expected_state_action_values[dim])
                    #print("loss_dim: ",loss_dim)
                    branch_losses.append(loss_dim)
                
                loss = sum(branch_losses) / self.num_branches

            else:
                # -----------------
                #   regular DQN
                # -----------------

                # get the Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = self.Q_target(obs_tp1).detach()
                q_tp1_values = self.divide(q_tp1_values, self.separate)
                q_s_a_prime = []
                a_prime = []
                loss = 0
                for dim in range(self.num_branches):
                    q_tp1_value = q_tp1_values[dim]
                    q_s_a_prime_dim, a_prime_dim = q_tp1_value.max(1)

                    q_s_a_prime_dim = (1 - done_mask) * q_s_a_prime_dim
                    q_s_a_prime.append(q_s_a_prime_dim)
                    a_prime.append(a_prime_dim)

                    expected_state_action_values_dim = rew_t + self.gamma * q_s_a_prime_dim

                    td_error_dim = q_s_a[dim] - expected_state_action_values_dim
                    if dim == 0:
                        td_error = torch.abs(td_error_dim)
                    else:
                        td_error += torch.abs(td_error_dim)

                    loss_dim = F.smooth_l1_loss(q_s_a[dim],expected_state_action_values_dim)
                    loss += loss_dim
                loss /= self.num_branches
            # clip the error and flip--old
            #clipped_error = -1.0 * error.clamp(-1, 1)

        # backwards pass
        self.optimizer.zero_grad()
        # if pre-train
        # TODO
        #if pre_train:
        #    dqloss = loss

        # TODO?
        #print("loss: ",loss)
        loss.backward()

        for param in self.Q.parameters():
            param.grad.data.clamp(-1,1)

        # update
        self.optimizer.step()
        self.num_param_updates += 1

        # update target Q nerwork weights with current Q network weights
        if self.num_param_updates % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
                
        return td_error, idxes
    
    def update_priorities(self, td_error, epsilon,idxes):
        if self.prioritized_replay:
            new_priorities = (torch.abs(td_error.detach()) + epsilon).squeeze()
            #print("new_priorities: ",new_priorities.size(),new_priorities)
            new_priorities = to_np(new_priorities)
            self.replay_buffer.update_priorities(idxes, new_priorities)
            #print("updata_success!")
        return 


    def log(self, ep, ep_rewards):
        if len(ep_rewards) > 0:
            self.mean_episode_reward = np.mean(ep_rewards[-20:])
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
        
        print('Timestep %d' %(self.t))
        print("learning started? %d" % (self.t >= self.learning_starts))
        print("num_param_updates:%d" % self.num_param_updates)
        print("mean reward (20 episodes) %f" % self.mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)
        print("exploration %f" % self.exploration.value(self.t))
        print("learning_rate %f" % self.optimizer_spec.kwargs['lr'])
        print("---------------------------------------")

        #=================== TensorBoard logging =============#
        # (1) Log the scalar values
        info = {
            'learning_started': (self.t > self.learning_starts),
            'num_episodes': ep,
            'exploration':self.exploration.value(self.t),
            'learning_rate': self.optimizer_spec.kwargs['lr'],
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, self.t+1)

        if len(ep_rewards) > 0:
            info = {
                'last_episode_rewards': ep_rewards[-1],
            }

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, self.t+1)

        if (self.best_mean_episode_reward != -float('inf')):
            info = {
                'mean_episode_reward_last_100': self.mean_episode_reward,
                'best_mean_episode_reward': self.best_mean_episode_reward
            }

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, self.t+1)

        if ep % self.Save_Reward_Every_N_EPs == 0:
            exploration = self.exploration.value(self.t)

            data = {"total_rewards":ep_rewards,
                        "exploration":exploration}

            if not os.path.exists("perform_records"):
                os.makedirs("perform_records")
    
            save_path = f"perform_records/ep{ep}.pkl"
            f = open(save_path,"wb")
            pickle.dump(data,f)
            f.close()

    def get_random_action(self):
        act = []
        for i in range(self.num_branches):
            num = self.num_actions_for_branch[i]
            a = np.random.randint(num)
            act.append(a)
        
        return act

    def get_action(self, act):
        """
        get action format to give to the env according to the act list []
        self.action_space [order of actions, camera at last]
        """
        action = self.env.action_space.noop()

        for idx in range(len(self.action_space)):
            a = self.action_space[idx]
            if(a == 'camera'):
                action[a] = [act[idx] * 10 - 180,act[idx+1] * 10 - 180]
            elif (a == 'place'):
                if(act[idx] == 0):
                    action[a] = 'none'
                else:
                    action[a] = 'dirt'
            else:
                action[a] = act[idx]
        
        return action

    def divide(self, x, separate_list):
        x_sep = []
        for i in range(len(separate_list)):
            if i == 0:
                cur = x[:,:separate_list[i]]
            else:
                cur = x[:,separate_list[i-1]:separate_list[i]]
            x_sep.append(cur)
        cur = x[:,separate_list[i]:]
        x_sep.append(cur)
        return x_sep

    def camera_transform(self,x):
        x = x + 180
        if (x >= 350):
            y = int(x / 10)
        else:
            y = round(x / 10)
        return y

    def store_demo_data(self):
        """
        Store demo data into the replay buffer.
        """
        pwd = os.getcwd()
        data_path = pwd + '/human_data'

        data = minerl.data.make(self.task, data_dir=data_path)
        data_dir = data_path + '/' + self.task
        streams = os.listdir(data_dir)

        total_frames = 0
        for stream in streams:
            #if(self.replay_buffer.num_in_buffer >= 1000):
                #break
            demo = data.load_data(stream, include_metadata=True)
            try:
                for current_state, action, reward, next_state, done, metadata in demo:
                    if metadata['success'] == False:
                        break
                    total_frames += 1
                    frame = current_state['pov']
                    compass = current_state['compassAngle']
                    dirt = current_state['inventory']['dirt']
                    non_pixel_feature = np.array([compass / 180.0, dirt / 64.0]).reshape(1, 2)
                    idx = self.replay_buffer.store_frame(frame, non_pixel_feature)

                    act = []
                    for a in self.action_space:
                        if a == 'camera':
                            act.append(self.camera_transform(action[a][0]))
                            act.append(self.camera_transform(action[a][1]))
                        else:
                            act.append(action[a])

                    reward = np.clip(reward, -1.0, 1.0)
                    done_int = 1 if done else 0
                    self.replay_buffer.store_effect(idx, np.array(act), reward, done_int)
            except RuntimeError:
                print(f"stream {stream} is corrupted!")
                continue

        assert total_frames == self.replay_buffer.num_in_buffer
        print(f"total frames {total_frames}")
        return total_frames

    def pre_train(self):
        """
        Pretrain phase.
        """
        print('Pre-training ...')
        for t in tqdm(range(self.pre_train_steps)):
            td_error, idxes = self.train(pre_train=True)
            # update priorities
            self.update_priorities(td_error, self.prioritized_replay_eps, idxes)
        print("All pre-train finish.")

        return


