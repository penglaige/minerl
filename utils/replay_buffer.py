# -*- coding: UTF-8 -*-

import numpy as np
import random
from utils.segment_tree import SumSegmentTree, MinSegmentTree

def sample_n_unique(sampling_f, n):
    '''Helper function. Given a function 'sampling_f' taht returns
    comparable objects, sample n such unique objects.
    '''
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer():
    def __init__(self, size, frame_history_len, non_pixel_dimension, add_non_pixel=False):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.non_pixel_dimension = non_pixel_dimension
        self.add_non_pixel = add_non_pixel

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs    = None
        self.action = None
        self.reward = None
        self.done   = None
        self.non_pixel_obs = None

    def can_sample(self, batch_size):
        '''Returns true if `batch_size` different transitions can be sampled from the buffer.'''
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self,idxes):
        #idxes are random numbers indicate the sampling index
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        if self.add_non_pixel:
            non_pixel_obs_batch = self.non_pixel_obs[idxes]
            next_non_pixel_obs_batch = self.non_pixel_obs[[idx + 1 for idx in idxes]]
            return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, non_pixel_obs_batch, next_non_pixel_obs_batch
        else:
            return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_non_pixel_feature(self):
        """Return the most recent non_pixel_feature.
        Returns
        -------
        non_pixel_feature: np.array
            Array of shape (1, non_pixel_dimension)
            and dtype np.uint8, 
        """
        assert self.num_in_buffer > 0
        return self.non_pixel_obs[(self.next_idx - 1) % self.size].reshape(1,self.non_pixel_dimension)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8, where observation[i*img_c:(i+1)*img_c, :, :]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        #将最后 frame_history_len 帧 frame 组合起来成为一个observation
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the lastes RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1): #除了最后一张
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0 ) # c, h, w instead of h, w, c
        else:
            # this optimazation has potential to saves about 30% compute time \o/
            # c, h, w instead of h, w, c
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame, non_pixel_feature):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # if observation is an image...
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
            if self.add_non_pixel:
                self.non_pixel_obs = np.empty([self.size, self.non_pixel_dimension],  dtype=np.float32)
        self.obs[self.next_idx] = frame
        if self.add_non_pixel:
            self.non_pixel_obs[self.next_idx] = non_pixel_feature

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret
    

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, frame_history_len, alpha, num_branches,non_pixel_dimension,add_non_pixel=False):
        """
        ----------
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size, frame_history_len,non_pixel_dimension,add_non_pixel)
        
        self.num_branches = num_branches

        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.num_in_buffer) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.num_in_buffer) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def store_frame(self, frame, non_pixel_feature):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # if observation is an image...
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size, self.num_branches],  dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
            if self.add_non_pixel:
                self.non_pixel_obs = np.empty([self.size, self.non_pixel_dimension],  dtype=np.float32)
        self.obs[self.next_idx] = frame
        if self.add_non_pixel:
            self.non_pixel_obs[self.next_idx] = non_pixel_feature

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret


    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0,self.num_in_buffer - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.num_in_buffer
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


#  demonstration replay buffer
class demoReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, size, frame_history_len, alpha, num_branches,non_pixel_dimension,add_non_pixel=False):
        super(demoReplayBuffer, self).__init__(size, frame_history_len, alpha, num_branches,non_pixel_dimension,add_non_pixel)

        self.demo_size = 0

    def store_frame(self, frame, non_pixel_feature):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # if observation is an image...
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size, self.num_branches],  dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
            if self.add_non_pixel:
                self.non_pixel_obs = np.empty([self.size, self.non_pixel_dimension],  dtype=np.float32)
        self.obs[self.next_idx] = frame
        if self.add_non_pixel:
            self.non_pixel_obs[self.next_idx] = non_pixel_feature

        ret = self.next_idx
        if self.next_idx >= self.size - 1:
            self.next_idx = (self.next_idx + 1) % self.size + self.demo_size
        else:
            self.next_idx += 1
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def n_step_sample(self,trajectory_n,idxes,gamma):
        # [batch, n, observation]
        n_reward_batch = []
        end_idxes = []
        for idx in idxes:
            r, end_idx = self._encode_n_step(trajectory_n, idx, gamma)
            n_reward_batch.append(r)
            end_idxes.append(end_idx)
    
        n_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in end_idxes], 0)
        n_reward_batch = np.array(n_reward_batch,dtype=np.float32)
        n_done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in end_idxes], dtype=np.float32)
        if self.add_non_pixel:
            n_non_pixel_obs_batch = self.non_pixel_obs[[idx + 1 for idx in end_idxes]]
            return n_obs_batch, n_reward_batch, n_non_pixel_obs_batch, n_done_mask
        else:
            return n_obs_batch, n_reward_batch,n_done_mask
        
    def _encode_n_step(self,trajectory_n, idx,gamma):
        start_idx = idx
        end_idx = start_idx
        for i in range(trajectory_n):
            end_idx += 1
            if self.done[end_idx % self.size]:
                break
    
        R = 0
        for t in range(end_idx-1,start_idx-1,-1):
            R = self.reward[t] + gamma * R
        return R, end_idx


        
