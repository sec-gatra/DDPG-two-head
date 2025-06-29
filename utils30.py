import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from env30 import GameState 

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
        )
        # two heads
        self.dist_head  = nn.Linear(net_width//4, action_dim)  # for softmax
        self.scale_head = nn.Linear(net_width//4, 1)           # for budget

        self.maxaction = maxaction

    def forward(self, state):
        x = self.net(state)                   # [B, hidden]
        logits = self.dist_head(x)            # [B, action_dim]
        dist   = F.softmax(logits, dim=-1)    # sum to 1

        scale  = torch.sigmoid(self.scale_head(x)).squeeze(-1)  
        # scale in (0,1), shape [B]

        total_power = scale * self.maxaction  # shape [B]
        # expand total_power to [B,action_dim] so we can multiply
        return dist * total_power.unsqueeze(-1)


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

def evaluate_policy(channel_gain, state, env, agent, turns=1):
    env = GameState(30,7)
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
    dr_rand11=0
    dr_rand12=0
    dr_rand13=0
    dr_rand14=0
    dr_rand15=0
    dr_rand16=0
    dr_rand17=0
    dr_rand18=0
    dr_rand19=0
    dr_rand20=0
    dr_rand21=0
    dr_rand22=0
    dr_rand23=0
    dr_rand24=0
    dr_rand25=0
    dr_rand26=0
    dr_rand27=0
    dr_rand28=0
    dr_rand29=0
    dr_rand30=0
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
    dr11=0
    dr12=0
    dr13=0
    dr14=0
    dr15=0
    dr16=0
    dr17=0
    dr18=0
    dr19=0
    dr20=0
    dr21=0
    dr22=0
    dr23=0
    dr24=0
    dr25=0
    dr26=0
    dr27=0
    dr28=0
    dr29=0
    dr30=0
    # threshold constraint (contoh)
    R_th = 0.048        # minimal data rate per UE [bit/s atau satuan yg kamu pakai]
    P_th = 15      # maksimal total power [W atau satuan yg kamu pakai]

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
            a_rand=env.sample_valid_power()
            

            # generate next state
            next_loc         = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, r, dw, tr, info = env.step(a, channel_gain, next_channel_gain)

            #step dari random 
            s_next1, r1, dw1, tr1, info1 = env.step(a_rand, channel_gain, next_channel_gain)
            print(f'DDPG power : {a}, reward :{r}, total power = {np.sum(a)}')
            print(f'random power : {a_rand}, reward :{r1}, total power = {np.sum(a_rand)}')

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
            dr11+=info['data_rate11']
            dr12+=info['data_rate12']
            dr13+=info['data_rate13']
            dr14+=info['data_rate14']
            dr15+=info['data_rate15']
            dr16+=info['data_rate16']
            dr17+=info['data_rate17']
            dr18+=info['data_rate18']
            dr19+=info['data_rate19']
            dr20+=info['data_rate20']
            dr21+=info['data_rate21']
            dr22+=info['data_rate22']
            dr23+=info['data_rate23']
            dr24+=info['data_rate24']
            dr25+=info['data_rate25']
            dr26+=info['data_rate26']
            dr27+=info['data_rate27']
            dr28+=info['data_rate28']
            dr29+=info['data_rate29']
            dr30+=info['data_rate30']
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
            dr_rand11+=info1['data_rate11']
            dr_rand12+=info1['data_rate12']
            dr_rand13+=info1['data_rate13']
            dr_rand14+=info1['data_rate14']
            dr_rand15+=info1['data_rate15']
            dr_rand16+=info1['data_rate16']
            dr_rand17+=info1['data_rate17']
            dr_rand18+=info1['data_rate18']
            dr_rand19+=info1['data_rate19']
            dr_rand20+=info1['data_rate20']
            dr_rand21+=info1['data_rate21']
            dr_rand22+=info1['data_rate22']
            dr_rand23+=info1['data_rate23']
            dr_rand24+=info1['data_rate24']
            dr_rand25+=info1['data_rate25']
            dr_rand26+=info1['data_rate26']
            dr_rand27+=info1['data_rate27']
            dr_rand28+=info1['data_rate28']
            dr_rand29+=info1['data_rate29']
            dr_rand30+=info1['data_rate30']
            

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
                info['data_rate11'],
                info['data_rate12'],
                info['data_rate13'],
                info['data_rate14'],
                info['data_rate15'],
                info['data_rate16'],
                info['data_rate17'],
                info['data_rate18'],
                info['data_rate19'],
                info['data_rate20'],
                info['data_rate21'],
                info['data_rate22'],
                info['data_rate23'],
                info['data_rate24'],
                info['data_rate25'],
                info['data_rate26'],
                info['data_rate27'],
                info['data_rate28'],
                info['data_rate29'],
                info['data_rate30'],
            ]
            if all(dr >= R_th for dr in data_rates):
                count_data_ok += 1

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
                info1['data_rate11'],
                info1['data_rate12'],
                info1['data_rate13'],
                info1['data_rate14'],
                info1['data_rate15'],
                info1['data_rate16'],
                info1['data_rate17'],
                info1['data_rate18'],
                info1['data_rate19'],
                info1['data_rate20'],
                info1['data_rate21'],
                info1['data_rate22'],
                info1['data_rate23'],
                info1['data_rate24'],
                info1['data_rate25'],
                info1['data_rate26'],
                info1['data_rate27'],
                info1['data_rate28'],
                info1['data_rate29'],
                info1['data_rate30'],
            ]
            if all(dr >= R_th for dr in data_rates1):
                count_data_ok_rand += 1
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
            total_power  += info['total_power']
            total_power_rand +=info1['total_power']

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
    avg_power_rand = total_power_rand / turns

    # hitung persentase constraint terpenuhi
    pct_data_ok  = 100 * count_data_ok  / total_steps
    pct_power_ok = 100 * count_power_ok / total_steps
    pct_data_ok_rand  = 100 * count_data_ok_rand  / total_steps
    pct_power_ok_rand = 100 * count_power_ok_rand / total_steps
    return {
        'avg_score':    avg_score,
        'avg_score_rand' : avg_score_rand,
        'avg_EE':       avg_EE,
        'avg_EE_rand':       avg_EE_rand,
        'avg_power':    avg_power,
        'avg_power_rand' : avg_power_rand,
        'pct_data_ok':  pct_data_ok,
        'pct_data_ok_rand':  pct_data_ok_rand,
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
        'data_rate_11': dr11,
        'data_rate_12': dr12,
        'data_rate_13': dr13,
        'data_rate_14': dr14,
        'data_rate_15': dr15,
        'data_rate_16': dr16,
        'data_rate_17': dr17,
        'data_rate_18': dr18,
        'data_rate_19': dr19,
        'data_rate_20': dr20,
        'data_rate_21': dr21,
        'data_rate_22': dr22,
        'data_rate_23': dr23,
        'data_rate_24': dr24,
        'data_rate_25': dr25,
        'data_rate_26': dr26,
        'data_rate_27': dr27,
        'data_rate_28': dr28,
        'data_rate_29': dr29,
        'data_rate_30': dr30,
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
        'data_rate_rand11': dr_rand11,
        'data_rate_rand12': dr_rand12,
        'data_rate_rand13': dr_rand13,
        'data_rate_rand14': dr_rand14,
        'data_rate_rand15': dr_rand15,
        'data_rate_rand16': dr_rand16,
        'data_rate_rand17': dr_rand17,
        'data_rate_rand18': dr_rand18,
        'data_rate_rand19': dr_rand19,
        'data_rate_rand20': dr_rand20,
        'data_rate_rand21': dr_rand21,
        'data_rate_rand22': dr_rand22,
        'data_rate_rand23': dr_rand23,
        'data_rate_rand24': dr_rand24,
        'data_rate_rand25': dr_rand25,
        'data_rate_rand26': dr_rand26,
        'data_rate_rand27': dr_rand27,
        'data_rate_rand28': dr_rand28,
        'data_rate_rand29': dr_rand29,
        'data_rate_rand30': dr_rand30,
        'data_rate_pass' : count, 
        'data_rate_rand_pass' : count_rand,
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
