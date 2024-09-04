from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
import gym
import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = 0
        
    def _on_step(self) -> bool:
        self.episode_rewards += self.locals["rewards"]
        
        for index in np.where(self.locals["dones"])[0]:
            self.rewards.append(self.episode_rewards[index])
            self.episode_rewards[index] = 0
        
        return True
    
models = {
    # "DQN": DQN,
    "A2C": A2C,
    # "PPO": PPO,
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cpu = 2
    
    for model_name, model_class in models.items():
        vec_env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(num_cpu)])
        model = model_class("MlpPolicy", vec_env)
        callback = CustomCallback()
        
        #training
        print(f"----{model_name}----")
        print("start training\ntraining...")
        start_time = time.time()
        model.learn(total_timesteps=100, callback=callback, progress_bar=True)

        best_reward = float("-inf")
        cumulative_reward = 0
        
        obs = vec_env.reset()
        done = np.array([False for _ in range(vec_env.num_envs)])
        
        #testing
        print("start evaluation")
        i = 0
        episode_reward = np.zeros(vec_env.num_envs)
        with tqdm(total=100) as pbar:
            while i < 100:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward
                for index in np.where(done)[0]:
                    if episode_reward[index] > best_reward:
                        best_reward = episode_reward[index]
                    cumulative_reward += episode_reward[index]
                    episode_reward[index] = 0
                    i += 1
                    pbar.update(1)
        
        #save results
        with open("SB3/cartpole/results.txt", 'a') as f:
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
        plt.title(f"CartPole-v1: {model_name}")
        plt.tight_layout()
        plt.savefig(f"./SB3/cartpole/graphs/CartPole_{model_name}.png")