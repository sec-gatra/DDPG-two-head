import numpy as np 
from scipy.spatial.distance import cdist
from typing import Optional

class GameState:
    def __init__(self, nodes, p_max, area_size=(20, 20)):
        self.nodes = nodes
        self.p_max = p_max
        self.gamma = 0.99
        self.gammal=0.05
        self.beta = 1
        self.noise_power = 0.01
        self.area_size = area_size
        self.positions = self.generate_positions()
        self.observation_space = 2*nodes * nodes + nodes  # interferensi, channel gain, power
        self.action_space = nodes
        self.p = np.random.uniform(0, self.p_max, size=self.nodes)
        self.rng = np.random.default_rng()
    def sample_valid_power(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)
        return rand * self.p_max
    def sample_valid_power2(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)  # jadi distribusi
        scale = np.random.uniform(0.0, 1.0)  # skala acak antara 0 dan 1
        return rand * (self.p_max) * scale


    def reset(self,gain,*, seed: Optional[int] = None, options: Optional[dict] = None):
        power = self.sample_valid_power()
        #super().ini(seed=seed)
        #loc = self.generate_positions()
        #gain= self.generate_channel_gain(loc)
        intr=self.interferensi(power,gain)
        #new_intr=self.interferensi_state(intr)
        #ini_sinr=self.hitung_sinr(ini_gain,intr,power)
        #ini_data_rate=self.hitung_data_rate(ini_sinr)
        #ini_EE=self.hitung_efisiensi_energi(self.p,ini_data_rate)
        gain_norm=self.norm(gain)
        intr_norm = self.norm(intr)
        p_norm=self.norm(power)
        
        result_array = np.concatenate((np.array(gain_norm).flatten(), np.array(intr_norm).flatten(),np.array(p_norm)))
        return result_array ,{}

    def step_function(self,x):
        if x<=0 :
            x= 0
        else :
            x=1
        return x
    def step(self,power,channel_gain,next_channel_gain):
        #a=channel_gain
        intr=self.interferensi(power,channel_gain)
        next_intr=self.interferensi(power,next_channel_gain)
        sinr=self.hitung_sinr(channel_gain,intr,power)
        data_rate=self.hitung_data_rate(sinr)
        count_data_ok = sum(1 for dr in data_rate if dr >= 0.152)
        data_rate_constraint=[]
        for i in range(self.nodes):
            data_rate_constraint.append(2*self.step_function(0.12-data_rate[i]))
            #data_rate_constraint.append(20*(data_rate[i]-4))
        EE=self.hitung_efisiensi_energi(power,data_rate)
        
        total_daya=np.sum(power)
        total_rate  = np.sum(data_rate)
        
        # Condition 1: Budget exceeded
        fail_power = total_daya > self.p_max

        rate_violation = np.sum(np.maximum(0.152 - data_rate, 0.0))
        penalty_rate   = rate_violation
        #print(f'channel gain {channel_gain}')
        print(f'data rate {data_rate}')
        print(f'EE {EE}')

        # 2) Power violation: only when total_power > p_max
        power_violation = max(0.0, total_daya - self.p_max)
        penalty_power   = 0.1 * power_violation

        # Reward: throughput minus penalties
        #reward = 0.1*EE  - 20*penalty_rate +10* total_rate #- penalty_power
        #reward = EE  - 5*penalty_rate - 0.5 * total_daya + total_rate
        reward = 10*total_rate -  total_daya - 2.0 * penalty_rate
        fairness_penalty = np.std(data_rate)  # biar agent ngasih alokasi lebih merata
        reward -= 5 * fairness_penalty


        # Final done flag for “dead/win”
        dw = bool(fail_power)

        info = {
        'EE': EE,
        'data_rate_pass' : count_data_ok,
        'data_rate1': data_rate[0],
        'data_rate2': data_rate[1],
        'data_rate3': data_rate[2],
        'data_rate4': data_rate[3],
        'data_rate5': data_rate[4],
        'data_rate6': data_rate[5],
        'data_rate7': data_rate[6],
        'data_rate8': data_rate[7],
        'data_rate9': data_rate[8],
        'data_rate10': data_rate[9],
        'total_power': float(np.sum(power))
        }
        #reward = -np.sum(data_rate_constraint) + EE - 1*self.step_function(total_daya-self.p_max)
        #if reward > 0 :
        #    reward += 3
        obs = np.concatenate([self.norm(next_channel_gain).ravel(),self.norm(next_intr).ravel(),self.norm(power)])
        return obs.astype(np.float32), float(reward), dw,False, info
    def norm(self,x):
        x = np.maximum(x, 1e-10) # aslinya kagak ada
        x_log = np.log10(x + 1e-10)  # +1e-10 untuk menghindari log(0)
        x_min = np.min(x_log)
        x_max = np.max(x_log)
        return (x_log - x_min) / (x_max - x_min + 1e-10) 
    
    def generate_positions(self, minDistance=2, subnet_radius=2, minD=0.5):
        rng = np.random.default_rng()
        bound = self.area_size[0] - 2 * subnet_radius

        X = np.zeros((self.nodes, 1), dtype=np.float64)
        Y = np.zeros((self.nodes, 1), dtype=np.float64)
        dist_2 = minDistance ** 2
        loop_terminate = 1
        nValid = 0

        while nValid < self.nodes and loop_terminate < 1e6:
            newX = bound * (rng.uniform() - 0.5)
            newY = bound * (rng.uniform() - 0.5)
            if all(np.greater(((X[0:nValid] - newX)**2 + (Y[0:nValid] - newY)**2), dist_2)):
                X[nValid] = newX
                Y[nValid] = newY
                nValid += 1
            loop_terminate += 1

        if nValid < self.nodes:
            print("Gagal menghasilkan semua controller dengan minDistance")
            return None

        # Geser ke koordinat positif di dalam area
        X = X + self.area_size[0] / 2
        Y = Y + self.area_size[0] / 2
        gwLoc = np.concatenate((X, Y), axis=1)

        # Buat posisi sensor di sekitar controllernya
        dist_rand = rng.uniform(low=minD, high=subnet_radius, size=(self.nodes, 1))
        angN = rng.uniform(low=0, high=2 * np.pi, size=(self.nodes, 1))
        D_XLoc = X + dist_rand * np.cos(angN)
        D_YLoc = Y + dist_rand * np.sin(angN)
        dvLoc = np.concatenate((D_XLoc, D_YLoc), axis=1)

        # Simpan posisi [controller, sensor] ke self.positions untuk dipakai jika perlu
        return cdist(gwLoc, dvLoc)
    '''
    def generate_channel_gain(self, dist, sigma_shadow_dB=7.0, frek=6, transmit_power=1, lambdA=0.05, plExponent=2.7):
        N = self.nodes

    # Convert frequency to wavelength if not provided
        if lambdA is None:
            c = 3e8  # speed of light
            lambdA = c / (frek * 1e9)  # Convert GHz to Hz

    # Shadow fading in dB
        S_dB = sigma_shadow_dB * np.random.standard_normal((N, N))
        S_linear = 10 ** (S_dB / 10)

        # Rayleigh fading (complex)
        h = (1 / np.sqrt(2)) * (
            np.random.standard_normal((N, N)) + 1j * np.random.standard_normal((N, N))
        )
    
        # Calculate channel gain (received power)
        power = (
            transmit_power
            * (4 * np.pi / lambdA) ** (-2)
            * (np.power(dist, -plExponent))
            * S_linear
            * np.abs(h) ** 2
        )
    
        return power
    '''
    def generate_channel_gain(self, dist, sigmaS=7.0, transmit_power=1.0, lambdA=0.05, plExponent=2.7):
        N = self.nodes
        S = sigmaS * self.rng.randn(N, N)
        S_linear = 10 ** (S / 10)

        h = (1 / np.sqrt(2)) * (self.rng.randn(N, N) + 1j * self.rng.randn(N, N))
        H_power = transmit_power * (4 * np.pi / lambdA) ** (-2) \
                  * np.power(dist, -plExponent) * S_linear * (np.abs(h) ** 2)
        return H_power
    '''
    def generate_channel_gain(self,dist, sigma_shadow_dB=7.0, frek = 6):
        N = self.nodes
        H = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                #if i != j :
                    PL_dB =  32.4  + 17.3* np.log10(dist[i, j]/1000)+20*np.log10(frek) #frek in GHz
                    shadowing_dB = np.random.normal(0, sigma_shadow_dB)
                    total_loss_dB = PL_dB + shadowing_dB

                    rayleigh_fading = np.abs(np.random.rayleigh(scale=1.0)) ** 2
                    H[i, j] = 10 ** (-total_loss_dB / 10) * rayleigh_fading
                #else :
                #    H[i, j] = np.abs(np.random.rayleigh(scale=1.0)) ** 2    
    
        return H
    '''
    def interferensi(self, power, channel_gain):
        interferensi = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    interferensi[i][j] = channel_gain[i][j] * power[j]  # ✅ FIXED
        return interferensi        
    
    def hitung_sinr(self, channel_gain, interferensi, power):
        sinr = np.zeros(self.nodes)
        for node_idx in range(self.nodes):
            desired = channel_gain[node_idx][node_idx] * power[node_idx]
            interference = np.sum(interferensi[node_idx])  # semua j ≠ i udah dihitung, [i][i] = 0
            sinr[node_idx] = desired / (interference + self.noise_power)
        return sinr


    def hitung_data_rate(self, sinr):
        sinr = np.maximum(sinr, 0)  # jika ada yang negatif, dibatasi 0
        return np.log2(1 + sinr)

    def hitung_efisiensi_energi(self, power, data_rate):
        """Menghitung efisiensi energi total"""
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        energi_efisiensi=total_rate / total_power if total_power > 0 else 0
        return energi_efisiensi
