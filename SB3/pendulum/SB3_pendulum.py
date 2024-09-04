from stable_baselines3 import DDPG, PPO, A2C, SAC, TD3
import gymnasium as gym
import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

class CustomCallback(BaseCallback):
    def __init__(self, n, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = [0 for _ in range(n)]
        
    def _on_step(self) -> bool:
        self.episode_rewards += self.locals["rewards"]
        
        if self.n_calls % 200 == 0:
            self.rewards.extend(self.episode_rewards)
            self.episode_rewards[:] = 0
        
        return True
    
models = {
    # "DDPG": {"model": DDPG, "params": {}, "timesteps": 25000, "num_cpu": 4},
    # "SAC": {"model": SAC, "params": {}, "timesteps": 25000, "num_cpu": 4},
    # "TD3": {"model": TD3, "params": {}, "timesteps": 25000, "num_cpu": 4},
    "A2C": {
        "model": A2C,
        "params": {},
        "timesteps": 200000,
        "num_cpu": 4
    },
    # "PPO": {"model": PPO, "params": {}, "timesteps": 300000, "num_cpu": 4},
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name, model_class in models.items():
        num_cpu = model_class["num_cpu"]
        vec_env = DummyVecEnv([lambda: gym.make("Pendulum-v1") for _ in range(num_cpu)])
        model = model_class["model"](policy="MlpPolicy", env=vec_env, **model_class["params"])
        callback = CustomCallback(n=num_cpu)
        
        if model_name == "A2C":
            vec_env = VecNormalize(vec_env)
        
        #training
        print(f"----{model_name}----")
        print("start training\ntraining...")
        start_time = time.time()
        model.learn(total_timesteps=model_class["timesteps"], callback=callback, progress_bar=True) #125 episodes

        obs = vec_env.reset()

        best_reward = float("-inf")
        cumulative_reward = 0

        #testing
        done = np.array([False for _ in range(vec_env.num_envs)])
        print("start evaluation")
        for i in tqdm(range(25)):
            episode_reward = 0
            done[:] = False
            while not done.all():
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward
                
            for er in episode_reward: 
                if er > best_reward:
                    best_reward = er,
            cumulative_reward += episode_reward.sum()
        
        #save results
        with open("SB3/pendulum/results.txt", 'a') as f:
            f.write(f"----{model_name}----\n")  
            f.write(f"Mean reward/episode: {cumulative_reward/100}\n")
            f.write(f"Best reward: {best_reward}\n")
            f.write(f"Execution time: {(time.time() - start_time)} seconds\n\n")
        
        #print results
        print(f"Mean reward/episode: {cumulative_reward/100}")
        print(f"Best reward: {best_reward}")
        print(f"Execution time: {(time.time() - start_time)} seconds\n")
        
        #plot training progress
        plt.figure()
        plt.plot(callback.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Pendulum-v1: {model_name}")
        plt.tight_layout()
        plt.savefig(f"./SB3/pendulum/graphs/Pendulum_{model_name}.png")