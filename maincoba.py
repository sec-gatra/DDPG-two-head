
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from envcoba import GameState
from ddpg import *
from collections import deque
import torch.nn as nn
import os, shutil
import argparse
from datetime import datetime
from utilscoba import str2bool,evaluate_policy


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default = 120000, help='Max training steps') #aslinya 5e6
parser.add_argument('--save_interval', type=int, default=2500, help='Model saving interval, in steps.') #aslinya 1e5
parser.add_argument('--eval_interval', type=int, default=2000, help='Model evaluating interval, in steps.') #aslinya 2e3

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=1024, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=5e-5, help='Learning rate of actor') # 2e-3
parser.add_argument('--c_lr', type=float, default=3e-6, help='Learning rate of critic') # 1e-3
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=25000, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.3, help='exploring noise') #aslinya 0.1
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device

def compute_cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def main():
    EnvName = ['Power Allocation','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    #BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    BrifEnvName = ['6G', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    
    # Build Env
    env = GameState(10,3)
    eval_env = GameState(10,3)
    opt.state_dim = env.observation_space
    opt.action_dim = env.action_space
    opt.max_action = env.p_max   #remark: action space【-max,max】
    #print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
    #      f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')

    #variable tambahan 
    iterasi = 200
    total_episode = -(-opt.Max_train_steps//iterasi)
    sepertiga_eps=total_episode//3
    EE_DDPG=[] #buat cdf
    EE_RAND=[] #buat_cdf
    RATE_SUCCESS=[]
    RATE_SUCCESS_RAND=[]
    POWER_DDPG = []
    POWER_RAND = []
    ALL_DATARATES_NODES = [[] for _ in range(env.nodes)]  # List terpisah untuk setiap node
    ALL_DATARATES=[]
    ALL_DATARATES_RAND=[]
    data_rate_1 =[]
    data_rate_4 =[]
    data_rate_7 =[]
    data_rate_10 =[]
    
    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
    #dummy_s = torch.zeros((1, opt.state_dim), device=opt.dvc)
    #print("Initial actor a:", agent.actor(dummy_s).cpu().detach().numpy())
    print("state_dim, action_dim =", opt.state_dim, opt.action_dim)
    #dummy = torch.zeros(1, opt.state_dim).to(opt.dvc)
    #print("actor out shape:", agent.actor(dummy).shape)
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        
        while True:
            loc = env.generate_positions()
            channel_gain=env.generate_channel_gain(loc)
            state_eval1,inf=env.reset(channel_gain)
            state_eval1 = np.array(state_eval1, dtype=np.float32)
            result = evaluate_policy(channel_gain,state_eval,eval_env, agent, turns=3)
            
            #print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score, )
    else:
        total_steps = 0
        lr_steps = 0
        while total_steps < opt.Max_train_steps: # ini loop episode. Jadi total episode adalah Max_train_steps/200
            lr_steps+=1
            if lr_steps==sepertiga_eps :
                opt.a_lr=0.3 * opt.a_lr
                opt.c_lr=0.3 * opt.c_lr
                opt.noise=opt.noise-0.1
                lr_steps=0
            loc= env.generate_positions() #lokasi untuk s_t
            channel_gain=env.generate_channel_gain(loc) #channel gain untuk s_t
            s,info= env.reset(channel_gain, seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False
            langkah = 0
            '''Interact & trian'''
            while not done: 
                #print(total_steps)
                langkah +=1
                if total_steps <= opt.random_steps: #aslinya < aja, ide pengubahan ini tuh supaya selec action di train dulu.
                    a = env.sample_valid_power()
                    #a = env.p
                else: 
                    a = agent.select_action(s, deterministic=False)
                next_loc= env.generate_positions() #lokasi untuk s_t
                next_channel_gain=env.generate_channel_gain(next_loc) #channel gain untuk s_t
                s_next, r, dw, tr, info= env.step(a,channel_gain,next_channel_gain) # dw: dead&win; tr: truncated
                writer.add_scalar("Reward iterasi", r, total_steps)

                loc= env.generate_positions()
                channel_gain=env.generate_channel_gain(loc)
                if langkah == iterasi :
                    tr= True
                  
                    
                done = (dw or tr)

                agent.replay_buffer.add(np.array(s, dtype=np.float32), a, r, np.array(s_next, dtype=np.float32), dw)
                s = s_next
                channel_gain=next_channel_gain
                total_steps += 1

                '''train'''
                if total_steps >= opt.random_steps:
                    a_loss, c_loss = agent.train()
                    writer.add_scalar("Loss/Actor", a_loss, total_steps)
                    writer.add_scalar("Loss/Critic", c_loss, total_steps)
                    # print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, actor_loss:{a_loss}')
                    # print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, c_loss:{c_loss}')
        
                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                #if total_steps == opt.Max_train_steps:
                    #st=0
                    #for i in range(200):
                    state_eval,inf=eval_env.reset(channel_gain)
                    state_eval = np.array(state_eval, dtype=np.float32)
                    result = evaluate_policy(channel_gain,state_eval,eval_env, agent, turns=1)
                    writer.add_scalar('reward_training', result['avg_score'], global_step=total_steps)
                    if total_steps == opt.Max_train_steps:
                        st=0
                        for i in range(15000):
                            loc_eval= env.generate_positions() #lokasi untuk s_t
                            channel_gain_eval=env.generate_channel_gain(loc_eval) #channel gain untuk s_t
                            state_eval,inf=eval_env.reset(channel_gain_eval)
                            state_eval = np.array(state_eval, dtype=np.float32)
                            result1 = evaluate_policy(channel_gain_eval,state_eval,eval_env, agent, turns=1)
                            data_rate_1.append(result1['data_rate_1'])
                            data_rate_4.append(result1['data_rate_4'])
                            data_rate_7.append(result1['data_rate_7'])
                            data_rate_10.append(result1['data_rate_10'])
                            for node_id in range(1, env.nodes+1):
                                ALL_DATARATES_NODES[node_id - 1].append(result1[f'data_rate_{node_id}'])
                                ALL_DATARATES.append(result1[f'data_rate_{node_id}'])
                                ALL_DATARATES_RAND.append(result1[f'data_rate_rand{node_id}'])
                            print(result1['avg_EE'])
                            print(result1['avg_EE_rand'])
                            EE_DDPG.append(result1['avg_EE'])
                            EE_RAND.append(result1['avg_EE_rand'])
                            RATE_SUCCESS.append(result1['pct_data_ok'])
                            RATE_SUCCESS_RAND.append(result1['pct_data_ok_rand'])
                            POWER_DDPG.append(result1['avg_power'])
                            POWER_RAND.append(result1['avg_power_rand'])
                            if opt.write: 
                                writer.add_scalar('ep_r', result1['avg_score'], global_step=st)
                                writer.add_scalar('energi efisiensi', result1['avg_EE'], global_step=st)
                                writer.add_scalar('energi efisiensi random', result1['avg_EE_rand'], global_step=st)
                                writer.add_scalar('total daya', result1['avg_power'], global_step=st)
                                writer.add_scalar('constraint daya', result1['pct_power_ok'], global_step=st)
                            print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(st)}')
                            st+=1
                        
                        

                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k')


                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
                s = s_next
                channel_gain=next_channel_gain

        x_ddpg, y_ddpg = compute_cdf(EE_DDPG)
        x_rand, y_rand = compute_cdf(EE_RAND)
        x_rate, y_rate = compute_cdf(RATE_SUCCESS)
        x_rate_rand, y_rate_rand = compute_cdf(RATE_SUCCESS_RAND)
        x_p, y_p = compute_cdf(POWER_DDPG)
        x_p_rand, y_p_rand = compute_cdf(POWER_RAND)
        
        # PLOT CDF EE
        fig, ax = plt.subplots()
        ax.plot(x_ddpg, y_ddpg, label='DDPG')
        ax.plot(x_rand, y_rand, label='Random')

        # Tambahan: panah horizontal untuk selisih di CDF = 0.5
        cdf_level = 0.5
        x1 = np.interp(cdf_level, y_ddpg, x_ddpg)
        x2 = np.interp(cdf_level, y_rand, x_rand)
        gap_percent = 100 * (x1 - x2) / x2

        ax.annotate(f"{gap_percent:.0f}%",
                    xy=((x1 + x2) / 2, cdf_level),
                    xytext=(x2, cdf_level + 0.05),
                    arrowprops=dict(arrowstyle='<->', color='black'),
                    ha='center', fontsize=11)
        ax.axhline(cdf_level, color='gray', linestyle=':', linewidth=1)

        ax.set_xlabel('Energi Efisiensi')
        ax.set_ylabel('CDF')
        ax.set_title('CDF Energi Efisiensi')
        ax.legend()
        ax.grid(True)
        fig.savefig("cdf_energy_efficiency.png", dpi=300)
        #     log figure
        if opt.write :
            writer.add_figure('CDF Energi Efisiensi', fig, global_step=st)
            plt.close(fig)

        # 2) Plot CDF Data Rate Success
        fig2, ax2 = plt.subplots()
        ax2.plot(x_rate, y_rate, label='DDPG')
        ax2.plot(x_rate_rand, y_rate_rand, label='Random')
        ax2.set_xlabel('Persentase UE ≥ R_th (%)')
        ax2.set_ylabel('CDF')
        ax2.set_title('CDF Success Rate Data Rate')
        ax2.legend()
        ax2.grid(True)

        if opt.write:
            writer.add_figure('CDF Data Rate Success', fig2, global_step=st)
            plt.close(fig2)

                # 3) Plot CDF power
        fig3, ax3 = plt.subplots()
        ax3.plot(x_p, y_p, label='Power DDPG')
        ax3.plot(x_p_rand, y_p_rand, label='Power Random')
        ax3.set_xlabel('Power')
        ax3.set_ylabel('CDF')
        ax3.set_title('CDF POWER')
        ax3.legend()
        ax3.grid(True)
        fig3.savefig("cdf_power.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Power', fig3, global_step=st)
            plt.close(fig3)
        # 4) Plot CDF Data Rate per Node
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        for idx, node_data in enumerate(ALL_DATARATES_NODES, 1):
            x, y = compute_cdf(node_data)
            ax4.plot(x, y, label=f'Node {idx}')

        # Garis vertikal untuk R_min
        R_min = 0.152
        ax4.axvline(R_min, color='red', linestyle='--', label=f'R_min = {R_min}')

        ax4.set_xlabel('Data Rate')
        ax4.set_ylabel('CDF')
        ax4.set_title('CDF of Data Rate per Node')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        ax4.grid(True)
        plt.tight_layout()
        fig4.savefig("cdf_node_rate.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Data Rate per Node', fig4, global_step=st)
            plt.close(fig4)
        # 5) Plot CDF Data Rate sistem
        x_dr, y_dr = compute_cdf(ALL_DATARATES)
        x_dr_rand, y_dr_rand = compute_cdf(ALL_DATARATES_RAND)
        fig5, ax5 = plt.subplots()
        ax5.plot(x_dr, y_dr, label='DDPG (All Nodes)')
        ax5.plot(x_dr_rand, y_dr_rand, label='Random (All Nodes)', linestyle='--')

        # Tambahkan garis vertikal R_min
        R_min = 0.152  # Ganti sesuai kebutuhan
        ax5.axvline(R_min, color='red', linestyle='--', label=f'R_min = {R_min}')

        # Tambahkan panah horizontal untuk menunjukkan gap di CDF 0.5
        cdf_level = 0.5
        x_ddpg_val = np.interp(cdf_level, y_dr, x_dr)
        x_rand_val = np.interp(cdf_level, y_dr_rand, x_dr_rand)
        gap_percent = 100 * (x_ddpg_val - x_rand_val) / x_rand_val

        ax5.annotate(f"{gap_percent:.0f}%",
                     xy=((x_ddpg_val + x_rand_val)/2, cdf_level),
                     xytext=(x_rand_val, cdf_level + 0.05),
                     arrowprops=dict(arrowstyle='<->', color='black'),
                     ha='center', fontsize=11)
        ax5.axhline(cdf_level, color='gray', linestyle=':', linewidth=1)

        ax5.set_xlabel('Data Rate')
        ax5.set_ylabel('CDF')
        ax5.set_title('CDF of Data Rate (All Nodes)')
        ax5.legend()
        ax5.grid(True)
        fig5.savefig("cdf_sistem_rate.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Data Rate Sistem', fig5, global_step=st)
            plt.close(fig5)

        # Buat dataframe
        df = pd.DataFrame({
            'EE_DDPG': EE_DDPG,
            'EE_RAND': EE_RAND,
            'data_rate_1' :data_rate_1,
            'data_rate_4' :data_rate_4,
            'data_rate_7' :data_rate_7,
            'data_rate_10' :data_rate_10,
            #'ALL_DATARATES' : ALL_DATARATES,
            'POWER_DDPG': POWER_DDPG,
            'POWER_RAND': POWER_RAND,
        })
        df1 = pd.DataFrame({
            'ALL_DATARATES' : ALL_DATARATES,
            'ALL_DATARATES_RANDOM' : ALL_DATARATES_RAND,
        })

# Simpan ke Excel
        df.to_excel(f'energi_efisiensi.xlsx', index=False)
        df1.to_excel(f'all_data_rate.xlsx', index=False)
        print(EE_DDPG)
        print(EE_RAND)
        print("The end")


#%load_ext tensorboard
#%tensorboard --logdir runs
if __name__ == '__main__':
    main()
