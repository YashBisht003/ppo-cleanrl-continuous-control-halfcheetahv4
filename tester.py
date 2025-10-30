import numpy as np
from dm_control import suite
from dm_control.viewer import launch
from stable_baselines3 import PPO

# Load the trained PPO model
model = PPO.load("ppo_cheetah_dmcontrol")


# Define a policy function compatible with dm_control.viewer
def policy(time_step):
    # Extract the observation from the time_step (a dict with multiple components)
    obs = time_step.observation
    # Flatten the observation dict into a single numpy array
    obs_flat = np.concatenate([v.flatten() for v in obs.values()])

    # Predict the action using the PPO model
    action, _ = model.predict(obs_flat, deterministic=True)

    return action


# Load the Half Cheetah environment from dm_control
env = suite.load(domain_name="cheetah", task_name="run")

# Launch the viewer with the custom policy
launch(env, policy=policy)
