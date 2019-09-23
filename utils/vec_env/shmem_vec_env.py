"""
An interface for asynchronous vectorized environments.
"""

import multiprocessing as mp
import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes
from baselines import logger

from .util import dict_to_obs, obs_space_info, obs_to_dict
from utils.utils import *
from utils.parser import *
import time

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool,
             np.int64: ctypes.c_int64}

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        # envs = [_thunk, _thunk]
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            logger.log('Creating dummy env object to get spaces')
            with logger.scoped_configure(format_strs=[]):
                dummy = env_fns[0]()
                observation_space, action_space = dummy.observation_space, dummy.action_space

                self.obs_space, self.act_space = observation_space, action_space
                self.pixel_shape, self.non_pixel_obs, self.non_pixel_input_size = parse_obs_space(self.obs_space)
                self.action_spaces, self.action_spaces_name = dddqn_parse_action_space(self.act_space)
                self.num_branches = len(self.action_spaces)
                self.action_template = dummy.action_space.noop()

                dummy.close()
                del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        # self.obs_keys: compass, inventory.dirt, pov
        # self.obs_shapes: (), (), (64,64,3)
        # self.obs_dtypes: float64, np.int64, ?
        #print("test 1:,\n", self.obs_keys,'\n',self.obs_shapes, '\n', self.obs_dtypes)
        #for k in self.obs_keys:
            #print("1:,",_NP_TO_CT[self.obs_dtypes[k].type])
            #print("2:,",int(np.prod(self.obs_shapes[k])))
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in env_fns]
        # self.obs_bufs[0] :{k: array(type, size)}
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for i in range(0,len(env_fns)):
                env_fn, obs_buf = env_fns[i], self.obs_bufs[i]
            #for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = NoDaemonProcess(target=_subproc_worker,
                            args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys))
                #proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        print("initial reset...")
        if self.waiting_step:
            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        # actions: numpy (num_process, num_branch)
        for pipe, act in zip(self.parent_pipes, actions):
            act = act.tolist()
            act = self.get_action(act)
            #print("act:",act)
            pipe.send(('step', act))
        self.waiting_step = True


    def step_wait(self):
        #print("time sleep...",)
        #time.sleep(0.01)
        print('step.....',end='')
        outs = [pipe.recv() for pipe in self.parent_pipes]
        #print("collection done...")
        print('done!')
        self.waiting_step = False
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def close(self):
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        #print("obs in: ", obs)
        result = {}
        for k in self.obs_keys:
            bufs = [b[k] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
            #print("o:",o)
            result[k] = np.array(o)
        return dict_to_obs(result)

    def get_action(self, act):
        """
        get action format to give to the env according to the act list []
        self.action_space [order of actions, camera at last]
        """
        action_template = self.action_template
        
        return get_actions_continuous(act, self.act_space, action_template)


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        #print("flatdict: ",flatdict)
        for k in keys:
            #print("k: ", k)
            dst = obs_bufs[k].get_obj()
            #print("dst: ", dst)
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
            sub_k = k.split('.')
            copy_obs = flatdict
            for i in range(len(sub_k)):
                copy_obs = copy_obs[sub_k[i]]
                #print("copy obs: ",copy_obs)
            #np.copyto(dst_np, flatdict[k])
            np.copyto(dst_np, copy_obs)
        #return dst_np

    env = env_fn_wrapper.x()
    #print("env, ",env)
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                step_done = False
                while not step_done:
                    try:
                        obs, reward, done, info = env.step(data)
                        step_done = True
                        #time.sleep(0.01)
                        if done:
                            #time.sleep(0.01)
                            print("reset....")
                            reset_done = False
                            while not reset_done:
                                try:
                                    obs = env.reset()
                                    reset_done = True
                                except:
                                    #print("remake the env!")
                                    #env.close()
                                    #env = env_fn_wrapper.x()
                                    print("....reset failed! Try again!")
                            #obs = env.reset()
                            print("reset done!")
                        pipe.send((_write_obs(obs), reward, done, info))
                    except:
                        print(".....step failed! Try again!!")
                        #env.close()
                        #env = env_fn_wrapper.x()
                        #obs = env.reset()
                        #print("Reset done!")
                        #pipe.send((_write_obs(obs), 0, True, None))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
