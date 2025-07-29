import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from envcoba import GameState 

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
'''
class Actor(nn.Module):
    """
    Deterministic actor that directly predicts per-node power allocations.
    Ensures p_i ≥ 0 and p_i ≤ Pmax individually, but does not enforce sum(p_i) ≤ Pmax.
    To add a sum-constraint, incorporate a soft penalty on sum(p) in your reward.
    """
    def __init__(self, state_dim: int, action_dim: int, net_width: int, Pmax: float):
        super().__init__()
        self.Pmax = Pmax

        # shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.LayerNorm(net_width),

            nn.Linear(net_width, net_width // 2),
            nn.ReLU(),
            nn.LayerNorm(net_width // 2),

            nn.Linear(net_width // 2, net_width // 4),
            nn.ReLU(),
            nn.LayerNorm(net_width // 4),
        )

        # head: predict raw positive power values
        self.out_head = nn.Linear(net_width // 4, action_dim)

        # initialize head weights small to avoid saturation
        nn.init.uniform_(self.out_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] input feature vector.

        Returns:
            p: [B, action_dim] with 0 <= p_ij <= Pmax.
        """
        x = self.net(state)                       # [B, net_width//4]
        raw_p = F.softplus(self.out_head(x))      # [B, action_dim], >= 0
        p = torch.clamp(raw_p, max=self.Pmax)     # enforce p_i <= Pmax
        return p
class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=1024):
        super().__init__()
        # pertama‐tama embed state saja
        self.l1 = nn.Linear(state_dim, net_width)
        self.ln1 = nn.LayerNorm(net_width)

        # setelah itu concat action, lalu dua layer lagi
        self.l2 = nn.Linear(net_width + action_dim, net_width//2)
        self.ln2 = nn.LayerNorm(net_width//2)

        self.l3 = nn.Linear(net_width//2, net_width//4)
        self.ln3 = nn.LayerNorm(net_width//4)

        # output Q‐value scalar
        self.l4 = nn.Linear(net_width//4, 1)

    def forward(self, state, action):
        """
        state:  Tensor [B, state_dim]
        action: Tensor [B, action_dim]
        """
        x = F.relu(self.ln1(self.l1(state)))             # [B, net_width]
        x = torch.cat([x, action], dim=-1)               # [B, net_width+action_dim]
        x = F.relu(self.ln2(self.l2(x)))                 # [B, net_width//2]
        x = F.relu(self.ln3(self.l3(x)))                 # [B, net_width//4]
        q = self.l4(x)                                   # [B, 1]
        return q
'''
def evaluate_policy_reward(channel_gain, state, env, agent, turns=3):
    total_reward = 0
    for j in range(turns):
        for i in range(200):
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True)
            next_loc = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, re, dw, tr, info = env.step(a, channel_gain, next_channel_gain)
            #if iterasi == max_iter :
            #    tr ==True
            done = (dw or tr)
            

            total_reward += re
            #iterasi +=1
            #print(i)
            state = s_next
            channel_gain = next_channel_gain
    return int(total_reward/3)
def evaluate_policy(channel_gain, state, env, agent, turns=1):
    env = GameState(10, 6)
    total_scores = 0
    total_scores_rand = 0 
    total_EE = 0
    total_EE_rand=0
    total_power = 0
    total_power_rand=0
    dr_rand1=0
    dr_rand2=0
    dr_rand3=0
    dr_rand4=0
    dr_rand5=0
    dr_rand6=0
    dr_rand7=0
    dr_rand8=0
    dr_rand9=0
    dr_rand10=0
    dr1=0
    dr2=0
    dr3=0
    dr4=0
    dr5=0
    dr6=0
    dr7=0
    dr8=0
    dr9=0
    dr10=0
    # threshold constraint (contoh)
    R_th = 0.354   # minimal data rate per UE [bit/s atau satuan yg kamu pakai]
    P_th = 6      # maksimal total power [W atau satuan yg kamu pakai]

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
    total_reward = 0
    max_iter=200
    channel_gain_reward = channel_gain
    '''
    for j in range(3):
        iterasi = 0
        done = False
        
        for i in range(200):
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True)
            next_loc = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, re, dw, tr, info = env.step(a, channel_gain_reward, next_channel_gain)
            if iterasi == max_iter :
                tr ==True
            done = (dw or tr)
            

            total_reward += re
            iterasi +=1
            #print(i)
            state = s_next
            channel_gain_reward = next_channel_gain
    '''

    for _ in range(turns):
        done = False
        MAX_STEPS = 1
        step_count = 0
        while not done:
            step_count += 1
            total_steps += 1

            # aksi deterministik
            a = agent.select_action(state, deterministic=True)
            

            #random_allocation 
            a_rand=env.sample_valid_power2()
            

            # generate next state
            next_loc         = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, r, dw, tr, info = env.step(a, channel_gain, next_channel_gain)

            #step dari random 
            s_next1, r1, dw1, tr1, info1 = env.step(a_rand, channel_gain, next_channel_gain)
            print(f'DDPG power : {a}, reward :{r}, total power : {np.sum(a)}')
            print(f'random power : {a_rand}, reward :{r1}, total power : {np.sum(a_rand)}')
            count_data_ok=info['data_rate_pass']
            count_data_ok_rand=info1['data_rate_pass']

            dr1+=info['data_rate1']
            dr2+=info['data_rate2']
            dr3+=info['data_rate3']
            dr4+=info['data_rate4']
            dr5+=info['data_rate5']
            dr6+=info['data_rate6']
            dr7+=info['data_rate7']
            dr8+=info['data_rate8']
            dr9+=info['data_rate9']
            dr10+=info['data_rate10']
            dr_rand1+=info1['data_rate1']
            dr_rand2+=info1['data_rate2']
            dr_rand3+=info1['data_rate3']
            dr_rand4+=info1['data_rate4']
            dr_rand5+=info1['data_rate5']
            dr_rand6+=info1['data_rate6']
            dr_rand7+=info1['data_rate7']
            dr_rand8+=info1['data_rate8']
            dr_rand9+=info1['data_rate9']
            dr_rand10+=info1['data_rate10']
      
            # cek constraint data rate: pastikan semua UE ≥ R_th untuk ddpg
            data_rates = [
                info['data_rate1'],
                info['data_rate2'],
                info['data_rate3'],
                info['data_rate4'],
                info['data_rate5'],
                info['data_rate6'],
                info['data_rate7'],
                info['data_rate8'],
                info['data_rate9'],
                info['data_rate10'],
            ]
            #count_data_ok = sum(1 for dr in data_rates if dr >= R_th)

            #cek data rate untuk random 
            data_rates1 = [
                info1['data_rate1'],
                info1['data_rate2'],
                info1['data_rate3'],
                info1['data_rate4'],
                info1['data_rate5'],
                info1['data_rate6'],
                info1['data_rate7'],
                info1['data_rate8'],
                info1['data_rate9'],
                info1['data_rate10'],
            ]
            #count_data_ok_rand = sum(1 for dr in data_rates1 if dr >= R_th)
            '''
            for i in range(env.nodes):
                if data_rates[i] >= R_th :
                    count+=1
                if data_rates1[i] >= R_th :
                    count_rand +=1
                jumlah_data_rate += data_rates[i]
                jumlah_data_rate_rand+=data_rates1[i]
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
            total_power  += np.sum(a)
            total_power_rand +=np.sum(a_rand)

            # update loop
            if step_count == MAX_STEPS:
                tr = True
            done = (dw or tr)
            state = s_next
            channel_gain = next_channel_gain

    # hitung rata-rata metrik
    avg_score = total_scores / turns
    avg_score_rand = total_scores_rand / turns
    avg_EE    = total_EE / turns
    avg_EE_rand = total_EE_rand / turns 
    avg_power = total_power / turns
    avg_power_rand = total_power_rand/turns

    # hitung persentase constraint terpenuhi
    pct_data_ok  = 100 * count_data_ok  / total_steps
    pct_power_ok = 100 * count_power_ok / total_steps
    pct_data_ok_rand  = 100 * count_data_ok_rand  / total_steps
    pct_power_ok_rand = 100 * count_power_ok_rand / total_steps
    return {
        'action' : a,
        'data_rate' : data_rates,
        'avg_score':    avg_score,
        'avg_score_rand' : avg_score_rand,
        'avg_EE':       avg_EE,
        'avg_EE_rand':       avg_EE_rand,
        'avg_power':    avg_power,
        'data_rate_lolos' : count_data_ok,
        'data_rate_lolos_rand' : count_data_ok_rand,
        'avg_power_rand' : avg_power_rand,
        'pct_data_ok':  count_data_ok,
        'pct_data_ok_rand':  count_data_ok_rand,
        'pct_power_ok_rand':  pct_power_ok_rand,
        'pct_power_ok': pct_power_ok,
        'data_rate_1': dr1,
        'data_rate_2': dr2,
        'data_rate_3': dr3,
        'data_rate_4': dr4,
        'data_rate_5': dr5,
        'data_rate_6': dr6,
        'data_rate_7': dr7,
        'data_rate_8': dr8,
        'data_rate_9': dr9,
        'data_rate_10': dr10,
        'data_rate_rand1': dr_rand1,
        'data_rate_rand2': dr_rand2,
        'data_rate_rand3': dr_rand3,
        'data_rate_rand4': dr_rand4,
        'data_rate_rand5': dr_rand5,
        'data_rate_rand6': dr_rand6,
        'data_rate_rand7': dr_rand7,
        'data_rate_rand8': dr_rand8,
        'data_rate_rand9': dr_rand9,
        'data_rate_rand10': dr_rand10,
        'data_rate_pass' : count, 
        'data_rate_rand_pass' : count_rand,
        'reward_train' : total_reward/3,
        #'data_rate_total' : jumlah_data_rate,
        #'data_rate_total_rand' : jumlah_data_rate_rand
        
        
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
