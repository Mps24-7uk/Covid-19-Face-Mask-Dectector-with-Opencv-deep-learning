import argparse
import contextlib

# Third-party imports
import gymnasium as gym
import numpy as np
import os
import time
import torch
import h5py

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")
parser.add_argument("--num_success_steps", type=int, default=10, help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def pre_process_actions(
    left_delta_pose, right_delta_pose, left_gripper, right_gripper,
    num_envs: int, device: str, expected_dim: int
) -> torch.Tensor:
    left_delta = torch.tensor(left_delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
    right_delta = torch.tensor(right_delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
    left_grip = torch.tensor([[1.0 if left_gripper else -1.0]], dtype=torch.float, device=device).repeat(num_envs, 1)
    right_grip = torch.tensor([[1.0 if right_gripper else -1.0]], dtype=torch.float, device=device).repeat(num_envs, 1)
    actions = torch.cat([left_delta, right_delta, left_grip, right_grip], dim=1)

    if actions.shape[1] < expected_dim:
        padding = torch.zeros((num_envs, expected_dim - actions.shape[1]), dtype=torch.float, device=device)
        actions = torch.cat([actions, padding], dim=1)
    return actions

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    keyboard = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)
    env.sim.reset()
    env.reset()
    keyboard.reset()

    success_count = 0
    current_demo = 0
    current_arm = "left"  # Toggle between 'left' and 'right'
    left_delta_pose = np.zeros(6)
    right_delta_pose = np.zeros(6)
    left_grip = False
    right_grip = False

    print("Press TAB to switch control between left and right arm.")

    # HDF5 recording setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.dirname(args_cli.dataset_file), exist_ok=True)
    hdf = h5py.File(args_cli.dataset_file.replace(".hdf5", f"_{timestamp}.hdf5"), "w")
    obs_grp = hdf.create_group("observations")
    act_grp = hdf.create_group("actions")
    frame = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            delta_pose, grip = keyboard.advance()

            if keyboard.is_key_pressed("tab"):
                current_arm = "right" if current_arm == "left" else "left"
                time.sleep(0.2)  # debounce to prevent rapid toggle

            if current_arm == "left":
                left_delta_pose = delta_pose
                left_grip = grip
            else:
                right_delta_pose = delta_pose
                right_grip = grip

            actions = pre_process_actions(
                left_delta_pose, right_delta_pose, left_grip, right_grip,
                env.num_envs, env.device, env.action_shape[1]
            )
            obs = env.step(actions)
            env.sim.render()

            # Save demo
            obs_grp.create_dataset(str(frame), data=obs["robot_joint_pos"].cpu().numpy())
            act_grp.create_dataset(str(frame), data=actions.cpu().numpy())
            frame += 1

            time.sleep(1.0 / args_cli.step_hz)

    hdf.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
