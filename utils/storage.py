import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils.parser import *

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def _faltten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage():
    def __init__(self, replay_buffer, frame_history_len, num_steps, num_processes, obs_space, act_space):
        """
        num_steps: int
            number of forward steps in A2C (default 5)
        num_processes: int
            number of processes
        obs_space: env.observation_space
        act_space: env.action_space
        """
        # parse obs_space and act_space
        self.pixel_shape, self.non_pixel_name, self.non_pixel_input_size = parse_obs_space(obs_space)
        # pixel_shape: tuple (h,w,c)
        self.pixel_shape = [self.pixel_shape[2] * frame_history_len, self.pixel_shape[0], self.pixel_shape[1]]
        self.action_spaces, self.action_spaces_name = dddqn_parse_action_space(act_space)
        self.num_branches = len(self.action_spaces)

        self.obs = torch.zeros(num_steps + 1, num_processes, *self.pixel_shape)
        if self.non_pixel_input_size > 0:
            self.non_pixel_obs = torch.zeros(num_steps + 1, num_processes, self.non_pixel_input_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, self.num_branches)

        self.actions = torch.zeros(num_steps, num_processes, self.num_branches)
        # ? long ?
        self.actions = self.actions.type(dtype)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        if self.non_pixel_input_size > 0:
            self.non_pixel_obs = self.non_pixel_obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, non_pixel_obs,actions, action_log_probs,
                value_preds, rewards, masks, bad_masks):
        """
        obs: torhc.tensor
            [num_processes,12, h, w]
        actions: torch.tensor
            [num_processes,num_branches]
        action_log_probs: torch.tensor
            [num_processes,num_branches]

        self.step   0   1   2   3   4   5   6
        obs             t
        actions     t
        action_log  t
        value_pred  t
        rewards     t
        returns     t
        masks           t
        bad_masks       t
        """
        self.obs[self.step + 1].copy_(obs)
        if self.non_pixel_input_size > 0:
            self.non_pixel_obs[self.step + 1].copy_(non_pixel_obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        if self.non_pixel_input_size > 0:
            self.non_pixel_obs[0].copy_(self.non_pixel_obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda,
                        use_proper_time_limits=True):
        """
        use_gae: 
            use generalized advantage estimation
        use_proper_time_limits: 
            compute returns taking into account time limits
        """
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                            step + 1] * self.masks[step + 1] - self.value_preds[step]
                    
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae

                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * gamma * \
                                    self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                    + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step + 1] - self.value_preds[step]
                    
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae

                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + \
                                        self.rewards[step]
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            non_pixel_obs_batch = None
            if self.non_pixel_input_size > 0:
                non_pixel_obs_batch = self.non_pixel_obs[:-1].view(-1,self.non_pixel_input_size)[indices]
            actions_batch = self.actions.view(-1,self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1,1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1,1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,self.actions.size(-1))[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1,1)[indices]

            yield obs_batch, non_pixel_obs_batch, actions_batch, value_preds_batch, return_batch, \
                    masks_batch, old_action_log_probs_batch, adv_targ

class NewRolloutStorage(RolloutStorage):
    def __init__(self, replay_buffer, frame_history_len, num_steps, num_processes, obs_space, act_space):
        super(NewRolloutStorage, self).__init__()

        self.temp_obs = torch.zeros(num_processes, num_steps, *self.pixel_shape)
        if self.non_pixel_input_size > 0:
            self.temp_non_pixel_obs = torch.zeros(num_processes, num_steps + 1, self.non_pixel_input_size)
        self.temp_rewards = torch.zeros(num_processes, num_steps, 1)
        self.temp_value_preds = torch.zeros(num_processes, num_steps + 1, 1)
        self.temp_actions = torch.zeros(num_processes， num_steps, self.num_branches)
        self.temp_action_log_probs = torch.zeros(num_processes, num_steps, self.num_branches)
        self.temp_actions = self.temp_actions.type(dtype)

        self.temp_masks = torch.ones(num_processes, num_steps + 1, 1)

        self.temp_bad_masks = torch.ones(num_processes, num_steps + 1, 1)

    def to(self, device):
        self.obs = self.obs.to(device)
        if self.non_pixel_input_size > 0:
            self.non_pixel_obs = self.non_pixel_obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

        self.temp_obs = self.temp_obs.to(device)
        if self.non_pixel_input_size > 0:
            self.temp_non_pixel_obs = self.temp_non_pixel_obs.to(device)
        self.temp_rewards = self.temp_rewards.to(device)
        self.temp_value_preds = self.temp_value_preds.to(device)
        self.returns = self.returns.to(device)
        self.temp_action_log_probs = self.temp_action_log_probs.to(device)
        self.temp_actions = self.temp_actions.to(device)
        self.temp_masks = self.temp_masks.to(device)
        self.temp_bad_masks = self.temp_bad_masks.to(device)

    def temp_insert(self, process, step, obs, non_pixel_obs,actions, action_log_probs,
                value_preds, rewards, masks, bad_masks):
        #obs: c h w
        self.temp_obs[process][step + 1].copy_(obs)
        if self.non_pixel_input_size > 0:
            self.temp_non_pixel_obs[process][step + 1].copy_(non_pixel_obs)
        self.temp_actions[process][step].copy_(actions)
        self.temp_action_log_probs[process][step].copy_(action_log_probs)
        self.temp_value_preds[process][step].copy_(value_preds)
        self.temp_rewards[process][step].copy_(rewards)
        self.temp_masks[process][step + 1].copy_(masks)
        self.temp_bad_masks[process][step + 1].copy_(bad_masks)

    
    def after_update(self):
        self.temp_obs[:,0].copy_(self.temp_obs[:,-1])
        if self.non_pixel_input_size > 0:
            self.temp_non_pixel_obs[:,0].copy_(self.temp_non_pixel_obs[:,-1])
        self.temp_masks[:,0].copy_(self.temp_masks[:,-1])
        self.temp_bad_masks[:,0].copy_(self.temp_bad_masks[:,-1])
        
    def _transpose(self):
        self.obs = self.temp_obs.transpose(1, 0, 2, 3, 4)
        if self.non_pixel_input_size > 0:
            self.non_pixel_obs = self.temp_non_pixel_obs.transpose(1, 0, 2)
        self.rewards = self.temp_rewards.transpose(1, 0, 2)
        self.value_preds = self.temp_value_preds.transpose(1, 0, 2)
        self.action_log_probs = self.temp_action_log_probs.transpose(1, 0 ,2)
        self.actions = self.temp_actions.transpose(1, 0, 2)
        self.masks = self.temp_masks.transpose(1, 0 ,2)
        self.bad_masks = self.temp_bad_masks.transpose(1, 0, 2)

