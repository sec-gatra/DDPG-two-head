
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
from utilscoba import str2bool,evaluate_policy, evaluate_policy_reward
import torch.nn.functional as F
import numpy as np
import torch
import copy
from torch.nn.utils import clip_grad_norm_


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default = 500000, help='Max training steps') #aslinya 5e6
parser.add_argument('--save_interval', type=int, default=2000, help='Model saving interval, in steps.') #aslinya 1e5
parser.add_argument('--eval_interval', type=int, default=500, help='Model evaluating interval, in steps.') #aslinya 2e3

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=1024, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor') # 2e-3
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic') # 1e-3
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=5000, help='random steps before trianing')#70000
parser.add_argument('--noise', type=float, default=0.05, help='exploring noise') #aslinya 0.1
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
    env = GameState(10,6)
    eval_env = GameState(10,6)
    opt.state_dim = env.observation_space
    opt.action_dim = env.action_space
    opt.max_action = env.p_max   

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
    CHANNEL_GAINS=[]
    reward=[]
    reward_rand=[]
    
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
        st=0
        channel_gains_from_csv1 = np.load('channel_gains_from_csv.npy', allow_pickle=True)
        #for i in range(3000):
        rate_lolos =[]
        rate_lolos_rand = []
        for i in range(len(channel_gains_from_csv1)):
                            st+=1
                            loc_eval= env.generate_positions() #lokasi untuk s_t
                            channel_gain_eval = channel_gains_from_csv1[i]
                            #channel_gain_eval=env.generate_channel_gain(loc_eval) #channel gain untuk s_t
                            state_eval,inf=eval_env.reset(channel_gain_eval)
                            state_eval = np.array(state_eval, dtype=np.float32)
                            result1 = evaluate_policy(channel_gain_eval,state_eval,eval_env, agent, turns=1)
                            rate_lolos.append(result1['data_rate_lolos'])
                            rate_lolos_rand.append(result1['data_rate_lolos_rand'])
                            for node_id in range(1, env.nodes+1):
                                ALL_DATARATES_NODES[node_id - 1].append(result1[f'data_rate_{node_id}'])
                                ALL_DATARATES.append(result1[f'data_rate_{node_id}'])
                                ALL_DATARATES_RAND.append(result1[f'data_rate_rand{node_id}'])
                            #print(result1['avg_EE'])
                            #print(result1['avg_EE_rand'])
                            #print(f'channel gain : {channel_gain_eval}')
                            #print(f'power  : {result1["action"]}')
                            #print(f'data rate : {result1["data_rate"]}')
                            #print(f'EE: {result1["avg_EE"]}')
                            CHANNEL_GAINS.append(channel_gain_eval.copy()) 
                            EE_DDPG.append(result1['avg_EE'])
                            EE_RAND.append(result1['avg_EE_rand'])
                            reward.append(result1['avg_score'])
                            reward_rand.append(result1['avg_score_rand'])
                            RATE_SUCCESS.append(result1['pct_data_ok'])
                            RATE_SUCCESS_RAND.append(result1['pct_data_ok_rand'])
                            POWER_DDPG.append(result1['avg_power'])
                            POWER_RAND.append(result1['avg_power_rand'])

                            #display the result
                            writer.add_scalar('ep_r', result1['avg_score'], global_step=st)
                            writer.add_scalar('energi efisiensi', result1['avg_EE'], global_step=st)
                            writer.add_scalar('energi efisiensi random', result1['avg_EE_rand'], global_step=st)
                            writer.add_scalar('total daya', result1['avg_power'], global_step=st)
                            writer.add_scalar('constraint daya', result1['pct_power_ok'], global_step=st)
        np.save('channel_gains.npy', np.array(CHANNEL_GAINS))
        x_ddpg, y_ddpg = compute_cdf(EE_DDPG)
        x_rand, y_rand = compute_cdf(EE_RAND)
        x_rate, y_rate = compute_cdf(RATE_SUCCESS)
        x_rate_rand, y_rate_rand = compute_cdf(RATE_SUCCESS_RAND)
        x_p, y_p = compute_cdf(POWER_DDPG)
        x_p_rand, y_p_rand = compute_cdf(POWER_RAND)
        
        # PLOT CDF EE
        fig, ax = plt.subplots()
        ax.plot(x_ddpg, y_ddpg, label='DDPG', linewidth=2.5)
        ax.plot(x_rand, y_rand, label='Random', linestyle='--', linewidth=2.5)
        ax.set_xlabel('Energy efficiency')
        ax.set_ylabel('CDF')
        ax.set_title('CDF Energy efficiency of 10 Nodes')
        ax.legend()
        ax.grid(False)  # Menghilangkan grid
        fig.savefig("cdf_energy_efficiency.png", dpi=300)
        #     log figure
        if opt.write :
            writer.add_figure('CDF Energi Efisiensi', fig, global_step=st)
            plt.close(fig)
                # PLOT CDF EE ddpg onlu
        fig0, ax0 = plt.subplots()
        ax0.plot(x_ddpg, y_ddpg, label='DDPG',linewidth=2.5, color = 'green')
        ax0.set_xlabel('Energy Efficiency')
        ax0.set_ylabel('CDF')
        ax0.set_title('CDF Energy Efficiency of DDPG (10 nodes)')
        ax0.legend()
        ax0.grid(False)  # Menghilangkan grid
        fig0.savefig("cdf_energy_efficiency_ddpg.png", dpi=300)
        #     log figure
        if opt.write :
            writer.add_figure('CDF Energi Efisiensi ddpg', fig0, global_step=st)
            plt.close(fig0)

        #plot cdf EE random only
        fig9, ax9 = plt.subplots()
        #ax9.plot(x_, y_ddpg, label='DDPG',linewidth=2.5)
        ax9.plot(x_rand, y_rand, label = "RANDOM", linewidth = 2.5, color = 'orange')
        ax9.set_xlabel('Energy Efficiency')
        ax9.set_ylabel('CDF')
        ax9.set_title('CDF Energy Efficiency of Random (10 nodes)')
        ax9.legend()
        ax9.grid(False)  # Menghilangkan grid
        fig9.savefig("cdf_energy_efficiency_random.png", dpi=300)
        #     log figure
        if opt.write :
            writer.add_figure('CDF Energi Efisiensi Random', fig9, global_step=st)
            plt.close(fig9)

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
        ax3.plot(x_p, y_p, label='Power DDPG',linewidth=2.5)
        ax3.plot(x_p_rand, y_p_rand, label='Power Random',linewidth=2.5, linestyle='--')
        ax3.set_xlabel('Power')
        ax3.set_ylabel('CDF')
        ax3.set_title('CDF POWER of 10 Nodes')
        ax3.legend()
        ax3.grid(False)  # Menghilangkan grid
        fig3.savefig("cdf_power.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Power', fig3, global_step=st)
            plt.close(fig3)

                # 3) Plot CDF power ddpg only
        figpd, axpd = plt.subplots()
        axpd.plot(x_p, y_p, label='Power DDPG',linewidth=2.5, color = 'green')
        #axpd.plot(x_p_rand, y_p_rand, label='Power Random',linewidth=2.5, linestyle='--')
        axpd.set_xlabel('Power')
        axpd.set_ylabel('CDF')
        axpd.set_title('CDF POWER DDPG of 10 Nodes')
        axpd.legend()
        axpd.grid(False)  # Menghilangkan grid
        figpd.savefig("cdf_power_ddpg.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Power ddpg', figpd, global_step=st)
            plt.close(figpd)

                # 4) Plot CDF power random only
        figpr, axpr = plt.subplots()
        #axpr.plot(x_p, y_p, label='Power DDPG',linewidth=2.5)
        axpr.plot(x_p_rand, y_p_rand, label='Power Random',linewidth=2.5, color = 'orange')
        axpr.set_xlabel('Power')
        axpr.set_ylabel('CDF')
        axpr.set_title('CDF POWER Random of 10 Nodes')
        axpr.legend()
        axpr.grid(False)  # Menghilangkan grid
        figpr.savefig("cdf_power_random.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Power random', figpr, global_step=st)
            plt.close(figpr)
        # 5) Plot CDF Data Rate per Node
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        for idx, node_data in enumerate(ALL_DATARATES_NODES, 1):
            x, y = compute_cdf(node_data)
            ax4.plot(x, y, label=f'Node {idx}')    
        # Garis vertikal untuk R_min
        R_min = 0.354
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
            
        # 6) Plot CDF Data Rate sistem
        x_dr, y_dr = compute_cdf(ALL_DATARATES)
        x_dr_rand, y_dr_rand = compute_cdf(ALL_DATARATES_RAND)
        fig5, ax5 = plt.subplots()
        ax5.plot(x_dr, y_dr, label='DDPG (All Nodes)')
        ax5.plot(x_dr_rand, y_dr_rand, label='Random (All Nodes)', linestyle='--')

        # Tambahkan garis vertikal R_min
        ax5.axvline(R_min, color='red', linestyle='--', label=f'R_min = {R_min}')

        ax5.set_xlabel('Data Rate')
        ax5.set_ylabel('CDF')
        ax5.set_title('CDF of Data Rate of 10 Nodes')
        ax5.legend()
        ax5.grid(False)
        fig5.savefig("cdf_sistem_rate.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Data Rate Sistem', fig5, global_step=st)
            plt.close(fig5)

                # 7) Plot CDF Data Rate sistem ddpg only
        #x_dr, y_dr = compute_cdf(ALL_DATARATES)
        #x_dr_rand, y_dr_rand = compute_cdf(ALL_DATARATES_RAND)
        figdr, axdr = plt.subplots()
        axdr.plot(x_dr, y_dr, label='DDPG (All Nodes)', color = 'green')
        #axde.plot(x_dr_rand, y_dr_rand, label='Random (All Nodes)', linestyle='--')

        # Tambahkan garis vertikal R_min
        axdr.axvline(R_min, color='red', linestyle='--', label=f'R_min = {R_min}')

        axdr.set_xlabel('Data Rate')
        axdr.set_ylabel('CDF')
        axdr.set_title('CDF of Data Rate DDPG of 10 Nodes')
        axdr.legend()
        axdr.grid(False)
        figdr.savefig("cdf_sistem_rate_DDPG.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Data Rate Sistem DDPG', figdr, global_step=st)
            plt.close(figdr)

        figdrr, axdrr = plt.subplots()
        #ax5.plot(x_dr, y_dr, label='DDPG (All Nodes)')
        axdrr.plot(x_dr_rand, y_dr_rand, label='Random (All Nodes)')

        # Tambahkan garis vertikal R_min
        axdrr.axvline(R_min, color='red', linestyle='--', label=f'R_min = {R_min}')

        axdrr.set_xlabel('Data Rate')
        axdrr.set_ylabel('CDF')
        axdrr.set_title('CDF of Data Rate Random of 10 Nodes')
        axdrr.legend()
        axdrr.grid(False)
        figdrr.savefig("cdf_sistem_rate_RANDOM.png", dpi=300)

        if opt.write:
            writer.add_figure('CDF Data Rate Sistem Random', figdrr, global_step=st)
            plt.close(figdrr)

                #data rate akurasi ddpg
        total_rate_lolos = np.sum(rate_lolos)
        total_node = env.nodes * 3000
        accuracy = total_rate_lolos * 100 / total_node
        print(f'accuracy data rate ddpg {accuracy}, maks node lolos per iterasi : {np.max(rate_lolos)}, min node lolos per iterasi : {np.min(rate_lolos)}')
        #data rate akurasi random
        total_rate_lolos_rand = np.sum(rate_lolos_rand)
        #total_node = env.nodes * 3000
        accuracy_rand = total_rate_lolos_rand * 100 / total_node
        print(f'accuracy data rate random {accuracy_rand}, maks node lolos per iterasi : {np.max(rate_lolos_rand)}, min node lolos per iterasi : {np.min(rate_lolos_rand)}')
        
        #totalenergi efisiensi 
        print(f'total energi efisiensi ddpg {np.sum(EE_DDPG)}')
        print(f'total energi efisiensi random {np.sum(EE_RAND)}')

                #total reward
        print(f'total reward ddpg {np.sum(reward)}')
        print(f'total reward random {np.sum(reward_rand)}')

        # Buat dataframe
        df = pd.DataFrame({
            'EE_DDPG': EE_DDPG,
            'EE_RAND': EE_RAND,
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
            
    else:
        total_steps = 0
        lr_steps = 0
        save=[]
        save2=[]
        save_from_critic =[]
        c_min = 9999
        ee=[]
        datret=[]
        last_model_path = None
        P = 1
        while total_steps < opt.Max_train_steps: # ini loop episode. Jadi total episode adalah Max_train_steps/200
            lr_steps+=1
            if total_steps%100000==0 :
                opt.a_lr=0.3 * opt.a_lr
                opt.c_lr=0.3 * opt.c_lr
                opt.noise=opt.noise-0.01
                lr_steps=0
            loc= env.generate_positions() #lokasi untuk s_t
            channel_gain=env.generate_channel_gain(loc) #channel gain untuk s_t
            s,info= env.reset(channel_gain, seed=env_seed)  
            env_seed += 1
            done = False
            langkah = 0
            '''Interact & trian'''
            while not done: 
                langkah +=1
                if total_steps <= opt.random_steps: 
                    a = env.sample_valid_power2()
                else: 
                    a = agent.select_action(s, deterministic=False)
                next_loc= env.generate_positions() 
                next_channel_gain=env.generate_channel_gain(next_loc) 
                s_next, r, dw, tr, info= env.step(a,channel_gain,next_channel_gain) 
                    
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
                    if c_loss <= c_min:
                        if last_model_path is not None and os.path.exists(last_model_path):
                            os.remove(last_model_path)
                
                        # simpan model baru
                        model_path = f"{BrifEnvName[opt.EnvIdex]}_{int(total_steps)}.pth"
                        agent.save(BrifEnvName[opt.EnvIdex], int(total_steps))
                
                        # update info terakhir
                        last_model_path = model_path
                        save_from_critic.clear()
                        save_from_critic.append(total_steps)
                        c_min = c_loss
                    with torch.no_grad():
                        s_batch, a_batch, _, _, _ = agent.replay_buffer.sample(opt.batch_size)
                        q_val = agent.q_critic(s_batch, a_batch).mean().item()
                        writer.add_scalar("Q_value/Mean", q_val, total_steps)

        
                '''record & log'''
                if total_steps % opt.eval_interval == 0:

                    print(f'learning rate actor : {opt.a_lr}')
                    print(f'learning rate critic : {opt.c_lr}')
                    state_eval,inf=eval_env.reset(channel_gain)
                    state_eval = np.array(state_eval, dtype=np.float32)
                    result = evaluate_policy(channel_gain,state_eval,eval_env, agent, turns=1)
                    result_reward = evaluate_policy_reward(channel_gain,state_eval,eval_env, agent, turns=3)
                    if result['avg_EE'] >= 100 and result['data_rate_lolos']>=0.8*env.nodes :
                        
                        agent.save(BrifEnvName[opt.EnvIdex], int(total_steps))
                        save2.append(int(total_steps))
                        
                    writer.add_scalar('reward_training', result["avg_score"], global_step=total_steps)
                    writer.add_scalar('reward_train', result["reward_train"], global_step=total_steps)
                    writer.add_scalar('reward training ddpg', result_reward, global_step=total_steps)
                    print(f'EE : {result["avg_EE"]}')
                    print(f'data rate lolos : {result["data_rate_lolos"]}')
                    print(f'steps : {total_steps}')


                    #print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, data rate : {result["pct_data_ok"]}')


                '''save model'''
               # if total_steps % opt.save_interval == 0:
               #     agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
                s = s_next
                channel_gain=next_channel_gain

       
        print("The end")
        print(save2)
        print(save_from_critic)


#%load_ext tensorboard
#%tensorboard --logdir runs
if __name__ == '__main__':
    main()
