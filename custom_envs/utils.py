from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN, DDPG, PPO, A2C, SAC, TD3

def normalize(x, min, max):
    return (x - min) / (max - min)

def unnormalize(x, min, max):
    return x * (max - min) + min

class CustomCallback(BaseCallback):
    def __init__(self, out_list, vec_env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.out_list = out_list
        self.env = vec_env

    def _on_step(self):
        self.out_list.append({
                # "temRoo.T": unnormalize(self.locals['new_obs']['temRoo_T'][0,0], 263.15, 343.15),
                "temRoo.T": self.locals['infos'][0]['temRoo.T'],
                "temSup.T": unnormalize(self.locals['new_obs']['temSup_T'][0,0], 273.15, 353.15),
                # "TOut.T": unnormalize(self.locals['new_obs']['TOut_T'][0,0], 253.15, 343.15),
                "heaPum.P": unnormalize(self.locals['new_obs']['heaPum_P'][0,0], 0, 100),
            }
        )
        
        return True
    
weathers = ["CH_BS_Basel", "CH_ZH_Maur", "CH_TI_Bellinzona", "CH_GR_Davos", "CH_GE_Geneva", "CH_VD_Lausanne"]

algorithms = {
    "DDPG": {"alg": DDPG,
        "alg_name": "DDPG",
        "params": {},
        "action_type": "continuous"
    },
    "SAC": {"alg": SAC,
        "alg_name": "SAC",
        "params": {},
        "action_type": "continuous"
    },
    "TD3": {"alg": TD3,
        "alg_name": "TD3",
        "params": {},
        "action_type": "continuous"
    },
    "A2C": {"alg": A2C,
        "alg_name": "A2C",
        "params": {},
        "action_type": "discrete"
    },
    "PPO": {"alg": PPO,
        "alg_name": "PPO",
        "params": {},
        "action_type": "discrete"
    },
    "DQN": {"alg": DQN,
        "alg_name": "DQN",
        "params": {},
        "action_type": "discrete"
    },
}