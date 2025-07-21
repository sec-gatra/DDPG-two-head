# main30.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil
import argparse
import torch
from datetime import datetime
from ddpg_new import DDPG_agent
from env30_new import GameState
from utils30_new import str2bool, evaluate_policy, evaluate_policy_reward

# Hyperparameter Setting
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIndex', type=int, default=0, help='0=6G Power Allocation env')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or not')
parser.add_argument('--ModelIndex', type=int, default=100, help='which model to load')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=30000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=2500, help='Model saving interval (steps)')
parser.add_argument('--eval_interval', type=int, default=2000, help='Model evaluation interval (steps)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--net_width', type=int, default=1024, help='hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-4, help='actor learning rate')
parser.add_argument('--c_lr', type=float, default=1e-3, help='critic learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--random_steps', type=int, default=5000, help='random steps before training')
parser.add_argument('--noise', type=float, default=0.2, help='initial exploration noise std')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)

# Utils
def compute_cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def main():
    # Set up environment and agent
    env = GameState(nodes=30, p_max=15)
    eval_env = GameState(nodes=30, p_max=15)
    opt.state_dim  = env.observation_space
    opt.action_dim = env.action_space
    opt.max_action = env.p_max

    # Seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    env.rng = np.random.default_rng(opt.seed)

    # Summary writer
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        now = datetime.now().strftime('%Y%m%d_%H%M')
        logdir = f"runs/PowerAlloc_{now}"
        if os.path.exists(logdir): shutil.rmtree(logdir)
        writer = SummaryWriter(log_dir=logdir)

    # Create model directory
    os.makedirs('model', exist_ok=True)
    agent = DDPG_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load('PowerAlloc', opt.ModelIndex)

    total_steps = 0
    episode = 0
    results = []

    while total_steps < opt.Max_train_steps:
        episode += 1
        # Reset env and noise
        loc = env.generate_positions()
        gain = env.generate_channel_gain(loc)
        state, a_prev, _ = env.reset(gain)
        agent.noise_proc.reset()
        done = False
        step_count = 0

        while not done and total_steps < opt.Max_train_steps:
            # Select action
            if total_steps < opt.random_steps:
                action = env.sample_valid_power2()
            else:
                action = agent.select_action(state)

            # Environment step
            next_loc = env.generate_positions()
            next_gain = env.generate_channel_gain(next_loc)
            next_state, reward, dw, tr, info = env.step(action, a_prev, gain, next_gain)

            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, dw)
            state, a_prev, gain = next_state, action, next_gain

            # Train
            if total_steps >= opt.random_steps:
                a_loss, q_loss = agent.train()
                if opt.write:
                    writer.add_scalar('loss/actor', a_loss, total_steps)
                    writer.add_scalar('loss/critic', q_loss, total_steps)

            total_steps += 1
            step_count += 1

            # Evaluate and save
            if total_steps % opt.eval_interval == 0:
                avg_metrics = evaluate_policy(gain, state, eval_env, agent, turns=1)
                avg_reward3 = evaluate_policy_reward(gain, state, eval_env, agent, turns=3)
                print(f"Step {total_steps}, Avg EE: {avg_metrics['avg_EE']:.2f}, Rate OK: {avg_metrics['data_rate_lolos']}")

                if avg_metrics['avg_EE'] >= 30 and avg_metrics['data_rate_pass'] >= 0.7 * env.nodes:
                    agent.save('PowerAlloc', total_steps)

                if opt.write:
                    writer.add_scalar('eval/avg_EE', avg_metrics['avg_EE'], total_steps)
                    writer.add_scalar('eval/EE_3runs', avg_reward3, total_steps)

            done = dw or tr

    print("Training completed.")

if __name__ == '__main__':
    main()
