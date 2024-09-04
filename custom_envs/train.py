import gym
import registration
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import CustomCallback, algorithms

if __name__ == "__main__":
    out_list = []
    num_cpu = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alg = algorithms['DQN']
    
    vec_env = SubprocVecEnv([lambda: gym.make("SwissHouseRSlaW2W-v0", action_type=alg['action_type']) for _ in range(num_cpu)])
    model = alg['alg']("MultiInputPolicy", vec_env, device=device)
    for _ in range(4):
        vec_env.reset()
        model.learn(total_timesteps=100000, callback=CustomCallback(out_list, vec_env), progress_bar=True)
      
    model.save(f"new_reward_function/{alg['alg_name']}_SwissHouseRSlaW2W")
    
    vec_env.close()
    out_df = pd.DataFrame(out_list)
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(12, 15))
    
    ax1.plot(out_df["temRoo.T"]-273.15, color='b', label='Room Temperature')
    # ax1.plot(out_df["temSup.T"]-273.15, color='g', label='Supply Temperature')
    ax1.axhline(y=16, color='r', linestyle='--')
    
    ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')

    plt.legend()
    plt.savefig(f"new_reward_function/{alg['alg_name']}_training.png")
    plt.show()