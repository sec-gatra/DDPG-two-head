import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from env30 import GameState 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super().__init__()
        # shared trunk
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.LayerNorm(net_width),
            nn.Linear(net_width, net_width//2),
            nn.ReLU(),
            nn.LayerNorm(net_width//2),
            nn.Linear(net_width//2, net_width//4),
            nn.ReLU(),
            nn.LayerNorm(net_width//4),
            nn.Linear(net_width//4, net_width//8),          # NEW LAYER 💥
            nn.ReLU(),
            nn.LayerNorm(net_width//8),
        )
        # two heads
        self.dist_head  = nn.Linear(net_width//8, action_dim)  # adjusted
        self.scale_head = nn.Linear(net_width//8, 1)           # adjusted

        self.maxaction = maxaction

    def forward(self, state):
        x = self.net(state)
        logits = self.dist_head(x)
        dist   = F.softmax(logits, dim=-1)

        scale  = torch.sigmoid(self.scale_head(x)).squeeze(-1)
        total_power = scale * self.maxaction
        return dist * total_power.unsqueeze(-1)



class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=1024):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.ln1 = nn.LayerNorm(net_width)

        self.l2 = nn.Linear(net_width + action_dim, net_width//2)
        self.ln2 = nn.LayerNorm(net_width//2)

        self.l3 = nn.Linear(net_width//2, net_width//4)
        self.ln3 = nn.LayerNorm(net_width//4)

        self.l4 = nn.Linear(net_width//4, net_width//8)         # NEW LAYER 💥
        self.ln4 = nn.LayerNorm(net_width//8)

        self.l5 = nn.Linear(net_width//8, 1)  # final Q-value

    def forward(self, state, action):
        x = F.relu(self.ln1(self.l1(state)))
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.ln2(self.l2(x)))
        x = F.relu(self.ln3(self.l3(x)))
        x = F.relu(self.ln4(self.l4(x)))                      # NEW LINE
        q = self.l5(x)
        return q

def evaluate_policy_reward(channel_gain, state, env, agent, turns=3):
    total_reward = 0
    for j in range(turns):
        a_prev=env.sample_valid_power2()
        for i in range(200):
            # Take deterministic actions at test time
           
            a = agent.select_action(state, deterministic=True)
            next_loc = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, re, dw, tr, info = env.step(a,a_prev, channel_gain, next_channel_gain)
            #if iterasi == max_iter :
            #    tr ==True
            done = (dw or tr)
            

            total_reward += re
            #iterasi +=1
            #print(i)
            a_prev=a
            state = s_next
            channel_gain = next_channel_gain
    return int(total_reward/3)

def evaluate_policy(channel_gain, state, env, agent, turns=1):
    env = GameState(30,15)
    total_scores = 0
    total_scores_rand = 0 
    total_EE = 0
    total_EE_rand=0
    total_power = 0
    total_power_rand=0
    # threshold constraint (contoh)
    R_th = 0.08     # minimal data rate per UE [bit/s atau satuan yg kamu pakai]
    P_th = 15  # maksimal total power [W atau satuan yg kamu pakai]

    # Counters untuk constraint
    count_data_ok  = 0
    count_data_ok_rand=0
    count_power_ok = 0
    count_power_ok_rand=0
    total_steps    = 0

    #counter data rate 
    count = 0 
    count_rand=0
    jumlah_data_rate = 0
    jumlah_data_rate_rand=0
    data_rate =[]
    data_rate_rand =[]
    total_rate_lolos=0
    rate_violation = 0

    for _ in range(turns):
        done = False
        MAX_STEPS = 1
        step_count = 0
        a_prev=env.sample_valid_power2()
        a_prev_rand=env.sample_valid_power2()
        while not done:
            step_count += 1
            total_steps += 1

            # aksi deterministik
            a = agent.select_action(state, deterministic=True)
            #random_allocation 
            a_rand=env.sample_valid_power()
            # generate next state
            next_loc         = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, r, dw, tr, info = env.step(a,a_prev, channel_gain, next_channel_gain)
            rate_violation = info['rate_violation']
            count_data_ok=info['data_rate_pass']
            #total_rate_lolos+=info['data_rate_pass']
            data_rate=info['data_rate']

            

            #step dari random 
            s_next1, r1, dw1, tr1, info1 = env.step(a_rand, a_prev_rand, channel_gain, next_channel_gain)
            print(f'DDPG power : {a}, reward :{r}, total power {np.sum(a)}')
            print(f'random power : {a_rand}, reward :{r1}, total power {np.sum(a_rand)}')
            data_rate_rand=info1['data_rate']
            '''
            for i in range(200):
                a = agent.select_action(state, deterministic=True)
                next_loc         = env.generate_positions()
                next_channel_gain= env.generate_channel_gain(next_loc)
                s_next, r, dw, tr, info = env.step(a,a_prev, channel_gain, next_channel_gain)
                total_rate_lolos+=info['data_rate_pass']
                #print(total_rate_lolos)
                state = s_next
                channel_gain = next_channel_gain
                a_prev=a
            '''
                
                
        
            # cek constraint power: total_power ≤ P_th
            if np.sum(a) <= P_th:
                count_power_ok += 1
            # cek constraint power: total_power ≤ P_th untuk random
            if np.sum(a_rand) <= P_th:
                count_power_ok_rand += 1

            # akumulasi reward & metrik lain
            total_scores += r
            total_scores_rand+=r1
            total_EE     += info['EE']
            total_EE_rand +=info1['EE']
            total_power  += info['total_power']
            total_power_rand +=info1['total_power']

            # update loop
            if step_count == MAX_STEPS:
                tr = True
            done = (dw or tr)
            state = s_next
            channel_gain = next_channel_gain
            a_prev=a
            a_prev_rand=a_rand

    # hitung rata-rata metrik
    avg_score = total_scores / turns
    avg_score_rand = total_scores_rand / turns
    avg_EE    = total_EE / turns
    avg_EE_rand = total_EE_rand / turns 
    avg_power = total_power / turns
    avg_power_rand = total_power_rand / turns
    total_rate_lolos=100*total_rate_lolos/(200*turns*30)
    return {
        'avg_score':    avg_score,
        'avg_score_rand' : avg_score_rand,
        'avg_EE':       avg_EE,
        'avg_EE_rand':       avg_EE_rand,
        'avg_power':    avg_power,
        'avg_power_rand' : avg_power_rand,
        'data_rate_lolos' : count_data_ok,
        'data_rate' : data_rate,
        'data_rate_rand' :data_rate_rand,
        'rate_violation' : rate_violation,
        #'total_rate_lolos' : total_rate_lolos,
        
    }

#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
