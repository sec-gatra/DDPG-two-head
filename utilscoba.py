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
'''
class Actori(nn.Module):#efficient fair
    def __init__(self, state_dim, action_dim, net_width, Pmax):
        super().__init__()
        self.Pmax = Pmax
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
        # head untuk distribusi efisiensi
        self.dist_head      = nn.Linear(net_width//4, action_dim)
        # head untuk total budget
        self.scale_head     = nn.Linear(net_width//4, 1)
        # head untuk trade-off fairness vs efficiency
        self.fairness_head  = nn.Linear(net_width//4, 1)

        # inisialisasi kecil untuk menghindari saturasi awal
        for head in (self.dist_head, self.scale_head, self.fairness_head):
            nn.init.uniform_(head.weight, -3e-3, 3e-3)
            nn.init.zeros_(head.bias)

    def forward(self, state):
        """
        state: [B, state_dim]
        returns p: [B, action_dim], with 0 <= p_i and sum_i p_i <= Pmax
        """
        x = self.net(state)                   # [B, H]
        # 1) base distribution (efficiency)
        logits = self.dist_head(x)            # [B, A]
        dist_eff = F.softmax(logits, dim=-1)  # sum=1

        # 2) total power scale
        scale = torch.sigmoid(self.scale_head(x)).squeeze(-1)  # [B] in (0,1)
        total_power = scale * self.Pmax                     # [B]

        # 3) fairness mixing coefficient
        alpha = torch.sigmoid(self.fairness_head(x)).squeeze(-1)  # [B] in (0,1)

        # 4) uniform dist
        A = dist_eff.size(-1)
        dist_uni = torch.full_like(dist_eff, 1.0/A)       # [B, A]

        # 5) final distribution: mix efficiency vs uniform
        dist_final = (1 - alpha).unsqueeze(-1)*dist_eff \
                     + alpha.unsqueeze(-1)*dist_uni       # [B, A], sum=1

        # 6) allocate power
        p = dist_final * total_power.unsqueeze(-1)        # [B, A], sum ≤ Pmax
        return p

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, Pmax):
        super().__init__()
        # shared trunk (sama seperti sebelumnya)
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
        # single head: raw power allocations
        self.out_head = nn.Linear(net_width//4, action_dim)
        self.Pmax = Pmax

    def forward(self, state):
        x = self.net(state)               # [B, net_width//4]
        raw_p = self.out_head(x)          # [B, action_dim]
        p_pos = F.relu(raw_p)             # ≥ 0
        p = self._project_simplex(p_pos)  # ensure sum ≤ Pmax
        return p

    def _project_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """
        Projects each row of v onto the simplex {p >= 0, sum(p) <= Pmax}.
        Uses the algorithm from:
        "Efficient Projections onto the l1-ball for Learning in High Dimensions" (Duchi et al., 2008).
        """
        # v: [B, n]
        B, n = v.size()
        # sort in descending order
        v_sorted, _ = torch.sort(v, descending=True, dim=-1)  # [B, n]
        cssv = v_sorted.cumsum(dim=-1)                       # cumulative sum
        arange = torch.arange(1, n+1, device=v.device).view(1, -1)
 #       # find the rho index for each batch
        cond = v_sorted - (cssv - self.Pmax) / arange > 0     # [B, n]
        rho = cond.sum(dim=-1) - 1                            # [B]
 #       # compute theta for each batch
        theta = (cssv[torch.arange(B), rho] - self.Pmax) / (rho + 1).float()  # [B]
 #       # project
        w = torch.clamp(v - theta.unsqueeze(-1), min=0.0)     # [B, n]
        return w
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

def evaluate_policy(channel_gain, state, env, agent, turns=1):
    env = GameState(10, 15)
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
    R_th = 0.152      # minimal data rate per UE [bit/s atau satuan yg kamu pakai]
    P_th = 3      # maksimal total power [W atau satuan yg kamu pakai]

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
            print(f'DDPG power : {a}, reward :{r}')
            print(f'random power : {a_rand}, reward :{r1}')

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
