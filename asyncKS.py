from time import sleep, time
from RLalgo.sac import SAC

from multiprocessing import Process, Manager
import torch
import numpy as np
from copy import deepcopy
from setproctitle import setproctitle

from envs import ks
from utility.RL import ReplayBuffer

class CacheBuffer(ReplayBuffer):
    def empty(self):
        self.obs_buf = 0*self.obs_buf
        self.obs2_buf = 0*self.obs2_buf
        self.act_buf = 0*self.act_buf
        self.rew_buf = 0*self.rew_buf
        self.done_buf = 0*self.done_buf
        self.ptr, self.size = 0, 0

    def copy(self, buf:ReplayBuffer):
        buf.store(self.obs_buf[:self.size].cuda(),
                  self.act_buf[:self.size].cuda(), 
                  self.rew_buf[:self.size].cuda(), 
                  self.obs2_buf[:self.size].cuda(), 
                  self.done_buf[:self.size].cuda(),self.size)

class autoKS(ks):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
    def run(self, Q_buffer, Cache_policy):
        self.reset()
        pre_state = self.state
        while True:
            if self.d or self.len>=self.maxt:
                self.reset()

            policy = Cache_policy[-1]
            if policy==None:
                action = self.action_space.sample()
            else:
                # policy.to('cpu').eval()
                action = policy.get_action(self.state)
            s,r,d,_ = self.step(action)
            Q_buffer.put((pre_state, action, r, s, d))


def test_RL(env,num_test_episode,max_len,RLNN,i=0,time=0):
    returnlist=[]
    RLNN.eval()
    RLNN.pi.act_len = RLNN.pi.act_len.to('cpu')
    o, ep_ret, ep_len = env.test_reset(num_test_episode), 0, 0
    while(ep_len < max_len):
        a = RLNN.get_action(torch.Tensor(o.squeeze(-1)), 0)
        o, r = env.step_p(o,a.numpy())
        ep_ret = r + ep_ret
        ep_len += 1
    returnlist=ep_ret
    return_=np.array(returnlist,dtype=np.float)
    
    mean,max,min=return_.mean().item(),return_.max().item(),return_.min().item()

    with open(ROOT,'a') as f:
        f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(i,mean,max,min,time))
    print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(i,mean,max,min,time))



class asyncSAC(SAC):
    def __call__(
        self, 
        epoch, 
        policy_after, 
        update_after, 
        update_every,
        batch_size, 
        test_every, 
        num_test_episodes, 
        update_iter,
        env: autoKS,
        max_async_iter = 100,
        more_RL_iter = 20000,
    ):

        test_env = env()
        envs = [env() for _ in range(NUM_ENVS)]
        manager = Manager()
        Q_buffer = manager.Queue()
        Cache_policy = manager.list([None])
          
        Processes = [Process(target=Env.run, args=(Q_buffer, Cache_policy),name='env'+str(i)) 
                        for Env,i in zip(envs,range(NUM_ENVS))]
        for p in Processes:
            setproctitle(p.name)
            p.start()
        start = time()
        
        RL_trained_flag = False
        last_test_size = 0
        tested = False
        last_buffer_size = 0
        cachebuffer = CacheBuffer(test_env.obs_dim,test_env.act_dim,int(2e5))
        async_iter = 1
        counter = 0
        itertime = []
        updatetime = []
        datatime = []
        nettime = []
        loop=0
        # main loop
        while self.buffer.size <= epoch:
            iterstart = time()
            if self.buffer.size > more_RL_iter:
                async_iter = max_async_iter
            # collect env data from workers
            if not Q_buffer.empty():
                # dtranstart = time()
                while not Q_buffer.empty():
                    cachebuffer.store(*Q_buffer.get())
                # datatime.append(time()-dtranstart)
                cachebuffer.copy(self.buffer)
                cachebuffer.empty()
            
            if self.buffer.size - last_buffer_size >= update_every:
                last_buffer_size = self.buffer.size
                counter = 0
                if self.buffer.size>=policy_after: 
                    netstart = time()
                    tosendpolicy = deepcopy(self.ac).cpu().eval()
                    tosendpolicy.pi.act_len = tosendpolicy.pi.act_len.to('cpu')
                    Cache_policy[0] = tosendpolicy
                    nettime.append(time()-netstart)


            if self.buffer.size>update_after and counter<async_iter:
                updatestart = time()
                for _ in range(update_iter):
                    batch = self.buffer.sample_batch(batch_size)
                    self.update(data=batch)
                updatetime.append([loop,time()-updatestart])
                RL_trained_flag=True
                counter+=1
                # print(counter)
            

            # # test & save
            if (RL_trained_flag and self.buffer.size-last_test_size>=test_every) or not tested:
                T = time()-start
                last_test_size = self.buffer.size
                tested = True
                RL_to_test = deepcopy(self.ac).to('cpu')
                RL_to_test.pi.act_len = RL_to_test.pi.act_len.to('cpu')
                testP = Process(target=test_RL, args=(test_env, num_test_episodes, 
                    self.max_ep_len, RL_to_test, self.buffer.size,T),name='Test')
                setproctitle(testP.name)
                testP.start()
                print('\nReal buffer size: {}\t'.format(self.buffer.size))
            
            sleep(0.1)
            
            itertime.append([loop,time()-iterstart])
            loop+=1
        np.save('time0',np.array(itertime))
        np.save('timeup',np.array(updatetime))
        np.save('datatrans',np.array(datatime))
        np.save('nettime',np.array(nettime))
        print('main exit')

if __name__=='__main__':
    import sys
    import os
    import random
    import torch.nn as nn
    from NN.RL.base import MLPQ
    from NN.RL.stochastic import GaussianMLPActor
    NUM_ENVS = sys.argv[1] #number of environments in parallel
    torch.set_num_threads(4) #change this accordingly https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    env = autoKS()
    ROOT = os.path.abspath(sys.argv[2]) # the saving path
    RLinp = { # Defining the RL accordingly, refer to the PIMBRL defination
        "env": env, 
        "Actor": GaussianMLPActor, 
        "Q": MLPQ,
        'a_kwargs':
            {
                "activation": nn.ReLU,
                "hidden_sizes": [256]*2,
                "output_activation": nn.Tanh
            }, 
        "act_space_type": "c",
        "gamma": 0.977,
        "device": "cuda"
    }
    RL = asyncSAC(**RLinp)
    with open(ROOT,'w+') as f:
        pass
    RL(6e3,0,1200,400,128,2000,2,50,autoKS,100) # more hyperparameters