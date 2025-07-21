# env30.py
import numpy as np
from scipy.spatial.distance import cdist

class GameState:
    def __init__(self, nodes, p_max, area_size=(20,20)):
        self.nodes = nodes
        self.p_max = p_max
        self.noise_power = 2e-10
        self.Rmin = 0.08
        self.area_size = area_size
        self.rng = np.random.default_rng()
        # Observation: [norm_gain, norm_interference, norm_power]
        self.observation_space = 2 * nodes * nodes + nodes
        self.action_space = nodes
    
    def sample_valid_power2(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)
        scale = np.random.uniform(0.0, 1.0)
        return rand * self.p_max * scale
    def sample_valid_power(self):
        return self.sample_valid_power2()

    def reset(self, gain, seed=None):
        power = self.sample_valid_power2()
        intr = self.interferensi(power, gain)
        sinr = self.hitung_sinr(gain, intr, power)
        data_rate = self.hitung_data_rate(sinr)
        obs = np.concatenate([
            self.norm(gain).ravel(),
            self.norm(intr).ravel(),
            self.norm(power)
        ])
        return obs.astype(np.float32), power, {}

    def step(self, power, power_prev, channel_gain, next_channel_gain):
        intr = self.interferensi(power_prev, channel_gain)
        sinr = self.hitung_sinr(channel_gain, intr, power)
        data_rate = self.hitung_data_rate(sinr)
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        rate_violation = np.sum(np.maximum(self.Rmin - data_rate, 0.0))

        penalty_rate  = rate_violation / self.nodes         # fraction of missing rate
        penalty_power = max(0.0, total_power - self.p_max) / self.p_max
        
        # trade‐off constants you can sweep over:
        α = 10.0     # penalty per unit of rate shortfall
        β =  1.0     # penalty per unit of power overshoot
        
        # reward = energy‐efficiency minus penalties
        #reward = (total_rate / total_power) - α * penalty_rate - β * penalty_power
        #data_rate_lolos=int(sum(dr >= self.Rmin for dr in data_rate))
        #frac_ok = data_rate_lolos / self.nodes
        #reward = (total_rate / total_power) * frac_ok
        
        total_power     = np.sum(power)
        total_rate      = np.sum(data_rate)
        n_ok            = int((data_rate >= self.Rmin).sum())
        frac_ok         = n_ok / self.nodes
        penalty_power   = max(0.0, total_power - self.p_max) / self.p_max
        
        # 2) energy‐efficiency with epsilon floor
        eps = 1e-2
        ee = total_rate / (total_power + eps)
        
        # 3) combine as a “soft” coverage‐weighted EE
        #   so if no users are covered, reward = 0 (not that huge EE)
        #   and as you cover more users, you harvest more of the EE incentive
        reward = frac_ok * ee \
               - 1.0 * penalty_power         # you can tune the 1.0 multiplier

# (no more hard −10 terms here)

        # Done flag: power budget violation ends episode
        dw = bool(total_power > self.p_max)
        info = {
            'EE': total_rate / total_power if total_power > 0 else 0.0,
            'data_rate_pass': int(sum(dr >= self.Rmin for dr in data_rate)),
            'total_power': float(total_power),
            'data_rate': data_rate,
            'rate_violation': float(rate_violation)
        }

        # Next observation
        intr_next = self.interferensi(power, next_channel_gain)
        obs_next = np.concatenate([
            self.norm(next_channel_gain).ravel(),
            self.norm(intr_next).ravel(),
            self.norm(power)
        ])
        return obs_next.astype(np.float32), float(reward), dw, False, info

    def norm(self, x):
        x = np.maximum(x, 1e-10)
        x_log = np.log10(x + 1e-10)
        return (x_log - x_log.min()) / (x_log.max() - x_log.min() + 1e-10)

    def interferensi(self, power, gain):
        I = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    I[i,j] = gain[i,j] * power[i]
        return I

    def hitung_sinr(self, gain, intr, power):
        sinr = np.zeros(self.nodes)
        for i in range(self.nodes):
            num = abs(gain[i,i]) * power[i]
            den = self.noise_power + np.sum(intr[i,:])
            sinr[i] = num / den
        return sinr

    def hitung_data_rate(self, sinr):
        return np.log1p(np.maximum(sinr, 0))

    def generate_positions(self, minDistance=2, subnet_radius=2, minD=0.5):
        # generate controller & device positions ensuring minDistance
        rng = self.rng
        bound = self.area_size[0] - 2 * subnet_radius
        X = np.zeros((self.nodes,1)); Y = np.zeros((self.nodes,1))
        dist_2 = minDistance**2; nValid = 0; loop=0
        while nValid < self.nodes and loop < 1e6:
            newX = bound * (rng.uniform() - 0.5)
            newY = bound * (rng.uniform() - 0.5)
            if all((X[:nValid,0]-newX)**2 + (Y[:nValid,0]-newY)**2 > dist_2):
                X[nValid,0] = newX; Y[nValid,0] = newY; nValid += 1
            loop += 1
        X += self.area_size[0]/2; Y += self.area_size[0]/2
        gw = np.concatenate((X,Y), axis=1)
        dist_rand = rng.uniform(minD, subnet_radius, (self.nodes,1))
        ang = rng.uniform(0, 2*np.pi, (self.nodes,1))
        dx = X + dist_rand*np.cos(ang)
        dy = Y + dist_rand*np.sin(ang)
        dv = np.concatenate((dx,dy), axis=1)
        return cdist(gw, dv)

    def generate_channel_gain(self, dist, sigmaS=7.0, transmit_power=1.0, lambdA=0.05, plExponent=2.7):
        # Log-normal shadowing & Rayleigh fading
        N = self.nodes
        S = sigmaS * self.rng.standard_normal((N,N))
        S_lin = 10 ** (S/10)
        real = self.rng.standard_normal((N,N)); imag = self.rng.standard_normal((N,N))
        h = (1/np.sqrt(2))*(real + 1j*imag)
        H_power = (transmit_power * (4*np.pi/lambdA)**(-2) * dist**(-plExponent) * S_lin * np.abs(h)**2)
        return H_power
