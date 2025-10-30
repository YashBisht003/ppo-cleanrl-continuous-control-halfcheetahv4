import gymnasium as gym
import numpy as np
from dm_control import suite
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from torch.optim.lr_scheduler import LambdaLR
import torch

# Set seed
SEED = 42

class DMCEnv(gym.Env):
    def __init__(self, domain_name, task_name,
                 normalize_observation=True, normalize_reward=True, seed=None):
        self.domain_name = domain_name
        self.task_name = task_name
        self._normalize_observation = normalize_observation
        self._normalize_reward = normalize_reward
        self.seed = seed

        # Load DM Control environment
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': seed})

        # Define action space
        spec = self.env.action_spec()
        self._action_space = spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

        # Define observation space
        obs_spec = self.env.observation_spec()
        flat_dim = sum(int(np.prod(obs_spec[k].shape)) for k in obs_spec)
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = seed
            self.env = suite.load(self.domain_name, self.task_name,
                                  task_kwargs={'random': seed})
        time_step = self.env.reset()
        obs = self._process_observation(time_step.observation)
        return obs, {}

    def step(self, action):
        action = np.clip(action, self._action_space.low, self._action_space.high)
        time_step = self.env.step(action)
        obs = self._process_observation(time_step.observation)
        reward = time_step.reward
        done = time_step.last()
        terminated = bool(done)
        truncated = False
        info = {}
        if self._normalize_reward:
            reward = self._normalize_reward_fn(reward)
        return obs, reward, terminated, truncated, info

    def _process_observation(self, obs_dict):
        obs = np.concatenate([obs_dict[k].flatten() for k in obs_dict])
        if self._normalize_observation:
            obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        return obs

    def _normalize_reward_fn(self, reward):
        return reward / (abs(reward) + 1e-8)

    def render(self, mode="human"):
        if mode == "human" or mode == "rgb_array":
            return self.env.physics.render()
        else:
            raise NotImplementedError

    def close(self):
        self.env.close()


# Helper to wrap the env in DummyVecEnv
def make_env(seed=SEED):
    def _init():
        env = DMCEnv(domain_name="cheetah", task_name="run",
                     normalize_observation=True, normalize_reward=True, seed=seed)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init

# Learning rate scheduler: linear decay from 3e-4 to 0 over training
def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

# Create vectorized environment
env = DummyVecEnv([make_env(seed=SEED)])

# Create PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=linear_schedule(3e-4),  # Linear LR schedule
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    seed=SEED,
)

# Train agent
model.learn(total_timesteps=1_000_000)

# Save model
model.save("ppo_cheetah_dmcontrol")
