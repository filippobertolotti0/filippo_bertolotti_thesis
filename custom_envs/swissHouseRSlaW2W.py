import energym
import gym
from gym import spaces
from utils import normalize, unnormalize
import numpy as np

class swissHouseRSlaW2W(gym.Env):    
    def __init__(self, action_type="continuous", weather="CH_BS_Basel"):
        self.env = energym.make('SwissHouseRSlaW2W-v0', weather=weather, simulation_days=365)
        self.total_temp_error = 0
        self.action_type = action_type
        self.set_point = 289.15
        self.last_action = 0
        
        if action_type == "discrete":
            self.action_space = spaces.Discrete(11)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        
        self.observation_space = spaces.Dict(
            {
                "heaPum_P": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "temSup_T": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "TOut_T": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                # "temRoo_T": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "delta": spaces.Box(low=0, high=1, shape=(1,), dtype=float)
            }
        )
        
    def set_set_point(self, set_point):
        self.set_point = set_point
                
    def _get_obs(self):
        outputs = self.env.get_output()
        room_temp = outputs["temRoo.T"]
        delta = self.set_point - room_temp
        
        return {
            "heaPum_P": np.array([normalize(outputs["heaPum.P"], 0, 100)], dtype=float),    #heat pump power
            "temSup_T": np.array([normalize(outputs["temSup.T"], 273.15, 353.15)], dtype=float),    #supply temperature
            "TOut_T": np.array([normalize(outputs["TOut.T"], 253.15, 343.15)], dtype=float),        #outdoor temperature
            "delta": np.array([delta], dtype=float)       #temperature delta
        }
                
    def reset(self, options=None, seed=None):
        self.env = energym.make('SwissHouseRSlaW2W-v0', simulation_days=365, eval_mode=True)
        obs = self._get_obs()

        return obs, {}
    
    def step(self, action):
        if self.action_type == "discrete":
            action = action * 0.1
        control = {'u': [action]}
        self.env.step(control)
        outputs = self._get_obs()
        reward = self.get_reward(outputs, action)[0]
        self.last_action = action
        info = dict({
            "temRoo.T": self.env.get_output()["temRoo.T"]
        })

        return outputs, reward, False, False, info
    
    def get_reward(self, outputs, action):
        delta = outputs["delta"]
        heat_pump_power = outputs["heaPum_P"]
        
        temp_error = -4 * abs(delta)
        energy_penalty = -0.001 * heat_pump_power
        reward = temp_error + energy_penalty
        
        return reward
    
    # def get_reward(self, outputs, action):
    #     delta = outputs["delta"]
    #     heat_pump_power = outputs["heaPum_P"]
        
    #     if abs(delta) < 1:
    #         smoothness_error = -abs(action - self.last_action)
    #     else:
    #         smoothness_error = 0
    #     temp_error = -2 * abs(delta)
    #     energy_penalty = -0.005 * heat_pump_power
    #     reward = temp_error + energy_penalty + smoothness_error
        
    #     return reward