import gym
import registration
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import normalize, unnormalize, weathers, algorithms
import numpy as np

if __name__ == "__main__":
    out_list = []
    num_cpu = 1
    alg = algorithms['DQN']

    vec_env = DummyVecEnv([lambda: gym.make("SwissHouseRSlaW2W-v0", action_type=alg['action_type']) for _ in range(num_cpu)])
    model = alg['alg'].load(f"new_reward_function/{alg['alg_name']}_SwissHouseRSlaW2W", env=vec_env)
    
    obs = vec_env.reset()
    steps = 4000
    cumulative_error = 0
    
    for i in tqdm(range(steps)):
        if i == 1000:
            vec_env.env_method('set_set_point', 294.15)
        elif i == 2000:
            vec_env.env_method('set_set_point', 292.15)
        if i == 3000:
            vec_env.env_method('set_set_point', 295.15)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)
        cumulative_error += abs(obs['delta'][0,0])
        out_list.append({
                "TOut.T": unnormalize(obs['TOut_T'][0,0], 253.15, 343.15),
                "temSup.T": unnormalize(obs['temSup_T'][0,0], 273.15, 353.15),
                "temRoo.T": info[0]['temRoo.T'],
                "heaPum.P": unnormalize(obs['heaPum_P'][0,0], 0, 100),
            }
        )
        
    vec_env.close()
    
    out_df = pd.DataFrame(out_list)
    print(f"Mean HeatPump power: {out_df['heaPum.P'].sum()/steps}")
    print(f"Mean temperature error: {cumulative_error/steps}")
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
    
    ax1.plot(out_df["temRoo.T"]-273.15, color='b', label='Room Temperature')
    ax1.plot(out_df["temSup.T"]-273.15, color='g', label='Supply Temperature')
    ax1.axhline(y=16, color='r', linestyle='--')
    ax1.axhline(y=19, color='r', linestyle='--')
    ax1.axhline(y=21, color='r', linestyle='--')
    ax1.axhline(y=22, color='r', linestyle='--')
    ax1.set_ylabel('Temp')
    ax1.set_xlabel('Steps')
    
    ax2.plot(out_df["heaPum.P"], color='b', label='Heat Pump Power')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Steps')
    
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.savefig(f"./new_reward_function/{alg['alg_name']}_test")
    plt.show()