
import numpy as np 
from scipy.spatial.distance import cdist
from typing import Optional

class GameState:
    def __init__(self, nodes, p_max, area_size=(20, 20)):
        self.nodes = nodes
        self.p_max = p_max
        self.gamma = 0.01
        self.beta = 1
        self.noise_power = 2e-10
        self.area_size = area_size
        self.positions = self.generate_positions()
        self.observation_space = 2*nodes * nodes + nodes  # interferensi, channel gain, power
        self.action_space = nodes
        self.p = np.random.uniform(0, 3, size=self.nodes)
        self.rng = np.random.default_rng()
        self.Rmin = 0.15
    def sample_valid_power(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)
        return rand * self.p_max
    def sample_valid_power2(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)  # jadi distribusi
        scale = np.random.uniform(0.0, 1.0)  # skala acak antara 0 dan 1
        return rand * (1) * scale

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
        intr=self.interferensi(power,channel_gain)
        next_intr=self.interferensi(power,next_channel_gain)
        sinr=self.hitung_sinr(channel_gain,intr,power)
        data_rate=self.hitung_data_rate(sinr)
        count_data_ok = sum(1 for dr in data_rate if dr >= self.Rmin)
        EE=self.hitung_efisiensi_energi(power,data_rate)
        
        total_daya=np.sum(power)
        total_rate  = np.sum(data_rate)
        
        # Condition 1: Budget exceeded
        fail_power = total_daya > self.p_max
        rate_violation = np.sum(np.maximum(self.Rmin - data_rate, 0.0))
        penalty_rate   = rate_violation
        #print(f'channel gain {channel_gain}')
        #print(f'data rate {data_rate}')
        #print(f'EE {EE}')
        # Parameter dinamis


        # 2) Power violation: only when total_power > p_max
        power_violation = max(0.0, total_daya - self.p_max)
        penalty_power   = 0.1 * power_violation
        k0 = 10           # Base penalty rate weight
        alpha = 0.1       # Semakin tinggi EE, semakin berat penalty rate
        beta = 0.5        # Penalti untuk total daya
        gammas = 1         # Penguat untuk sum-rate
        
        # Koefisien penalty rate tergantung EE
        k_dynamic = k0 + alpha * EE
        #fairness_penalty = np.std(data_rate)
        # Reward formula dinamis
 
        reward = 10*np.log(EE) - rate_violation - power_violation

        # Condition 2: Any data rate below threshold
        #min_rate = 0.5
        #fail_rate = np.any(data_rate < min_rate)

        # Final done flag for “dead/win”
        dw = bool(fail_power)

        info = {
        'EE': EE,
        'data_rate_pass' : count_data_ok,
        'total_power': float(np.sum(power)),
        'data_rate' : data_rate,
        'total_rate' : total_rate,
        }

        #reward = -np.sum(data_rate_constraint) + EE - 5*self.step_function(total_daya-self.p_max)
        obs = np.concatenate([self.norm(next_channel_gain).ravel(),self.norm(next_intr).ravel(),self.norm(power)])
        return obs.astype(np.float32), float(reward), dw,False, info
    def norm(self,x):
        x = np.maximum(x, 1e-10) # aslinya kagak ada
        x_log = np.log10(x + 1e-10)  # +1e-10 untuk menghindari log(0)
        x_min = np.min(x_log)
        x_max = np.max(x_log)
        return (x_log - x_min) / (x_max - x_min + 1e-10) 

    #def generate_positions(self):
    #    """Generate random positions for all nodes in 2D space (meter)"""
    #    loc = np.random.uniform(0, self.area_size[0], size=(self.nodes, self.nodes))
    #    for i in range (self.nodes) :
    #        for j in range (self.nodes):
    #          current = loc[i][j]
    #          loc[j][i]=current
    #    return loc
    
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
    def generate_channel_gain(self, dist, sigmaS=7.0, transmit_power=1.0, lambdA=0.05, plExponent=2.7):
        N = self.nodes
    
        # Shadowing (log-normal in dB scale)
        S = sigmaS * self.rng.standard_normal((N, N))
        S_linear = 10 ** (S / 10)
    
        # Rayleigh fading (complex): use standard_normal instead of randn
        real = self.rng.standard_normal((N, N))
        imag = self.rng.standard_normal((N, N))
        h = (1 / np.sqrt(2)) * (real + 1j * imag)
    
        # Compute channel gain (H_power)
        H_power = (
            transmit_power
            * (4 * np.pi / lambdA) ** (-2)
            * np.power(dist, -plExponent)
            * S_linear
            * np.abs(h) ** 2
        )
    
        return H_power

    def interferensi(self, power,channel_gain):
        interferensi = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    interferensi[i][j] = channel_gain[i][j] * power [i]
                else:
                    interferensi[i][j] = 0
        return interferensi
    
    def interferensi_state(self, interferensi):
        interferensi_state = np.zeros(self.nodes)
        for i in range(self.nodes):
            for j in range(self.nodes):
                interferensi_state[i]+=interferensi[j][i]
        return interferensi_state
        
    def hitung_sinr(self, channel_gain, interferensi, power):
        sinr = np.zeros(self.nodes)
        for node_idx in range(self.nodes):
            sinr_numerator = (abs(channel_gain[node_idx][node_idx])) * power[node_idx]
            sinr_denominator = self.noise_power + np.sum([(abs(interferensi[node_idx][i])) for i in range(self.nodes) if i != node_idx]) #aslinya noise_power**2
            sinr[node_idx] = sinr_numerator / sinr_denominator
        return sinr 

    def hitung_data_rate(self, sinr):
        sinr = np.maximum(sinr, 0)  # jika ada yang negatif, dibatasi 0
        return np.log(1 + sinr)

    def hitung_efisiensi_energi(self, power, data_rate):
        """Menghitung efisiensi energi total"""
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        energi_efisiensi=total_rate / total_power if total_power > 0 else 0
        return energi_efisiensi
