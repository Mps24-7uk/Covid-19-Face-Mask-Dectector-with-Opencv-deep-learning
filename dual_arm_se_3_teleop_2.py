from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import gymnasium as gym
import torch

# 1. Load env
env_cfg = parse_env_cfg("Isaac-Franka-PickPlace-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-Franka-PickPlace-v0", cfg=env_cfg).unwrapped
env.reset()

# 2. Init keyboard
keyboard = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)

# 3. Loop
while True:
    delta_pose, gripper_state = keyboard.advance()

    # 4. Prepare action
    action = torch.tensor([*delta_pose, 1.0 if gripper_state else -1.0], dtype=torch.float32).unsqueeze(0).to(env.device)

    # 5. Step
    obs, _, _, _ = env.step(action)

    # 6. (Optional) render
    env.sim.render()
