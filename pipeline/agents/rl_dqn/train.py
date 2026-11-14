"""Deep RL: DQN (stable-baselines3) sobre un entorn gymnasium que reprodueix la física oficial.
Entrenament offline amb torch; la xarxa Q s'exporta a numpy (qnet.npz) perquè el servidor no té torch.
Episodis = finestres aleatòries de 7 dies de 'fit' amb SOC inicial aleatori.
Ús: python pipeline/agents/rl_dqn/train.py [timesteps]"""
import sys
from pathlib import Path
import numpy as np, gymnasium as gym
from stable_baselines3 import DQN
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
from common import ACTIONS, wvec, feat, clamp, step_soc
from data import load_hourly, split_val

fit, _ = split_val(load_hourly())
co2 = fit.co2_kg_per_kwh.values; pr = fit.precio_eur_per_kwh.values
W = [wvec(co2[t:t + 24], pr[t:t + 24]) for t in range(len(fit))]


class Env(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (27,), np.float32)
    action_space = gym.spaces.Discrete(len(ACTIONS))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = int(self.np_random.integers(0, len(W) - 169)); self.end = self.t + 168
        self.soc = float(self.np_random.uniform(0, 2000))
        return feat(self.soc, W[self.t]).astype(np.float32), {}

    def step(self, a):
        p = clamp(self.soc, ACTIONS[a])
        r = -W[self.t][0] * p / 50.0
        self.soc = step_soc(self.soc, p); self.t += 1
        done = self.t >= self.end
        return feat(self.soc, W[self.t]).astype(np.float32), r, done, False, {}


steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
model = DQN("MlpPolicy", Env(), learning_rate=5e-4, buffer_size=200_000, learning_starts=5_000,
            batch_size=256, gamma=0.97, target_update_interval=2_000, exploration_fraction=0.3,
            exploration_final_eps=0.02, policy_kwargs=dict(net_arch=[128, 128]), seed=0, verbose=0)
model.learn(total_timesteps=steps, progress_bar=False)
# export a numpy
sd = model.q_net.q_net.state_dict()
arrs = {k.replace(".", "_"): v.cpu().numpy() for k, v in sd.items()}
np.savez(HERE / "qnet.npz", **arrs)
print("exportat", list(arrs))
