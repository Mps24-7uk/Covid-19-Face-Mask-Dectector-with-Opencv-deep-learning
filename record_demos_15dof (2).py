
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import time
import contextlib
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# -- CLI ARGUMENTS --
parser = argparse.ArgumentParser(description="Record 15-DOF demos via keyboard teleop.")
parser.add_argument("--task", type=str, required=True, help="Name of the IsaacLab task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset_15dof.hdf5")
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--num_demos", type=int, default=0)
parser.add_argument("--teleop_device", type=str, default="keyboard_15-DOF")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imports that require simulator
from custom_teleop_controller_15dof import CustomTeleopController15DOF
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RateLimiter:
    def __init__(self, hz): self.hz = hz; self.sleep_dur = 1.0 / hz; self.last = time.time()
    def sleep(self, env): 
        while time.time() < self.last + self.sleep_dur: 
            time.sleep(min(0.033, self.sleep_dur)); env.sim.render()
        self.last += self.sleep_dur


def pre_process_actions(action: np.ndarray, num_envs: int, device: str) -> torch.Tensor:
    return torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)


def create_teleop_device(device_name: str):
    if device_name == "keyboard_15-DOF":
        return CustomTeleopController15DOF(pos_sensitivity=0.2, rot_sensitivity=0.5)
    raise ValueError(f"Unsupported device: {device_name}")


def main():
    output_dir = os.path.dirname(args.dataset_file)
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = os.path.basename(args.dataset_file)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    teleop = create_teleop_device(args.teleop_device)

    env.sim.reset()
    env.reset()
    rate = RateLimiter(args.step_hz)

    demo_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            action = teleop.advance()
            torch_action = pre_process_actions(action, env.num_envs, env.device)
            env.step(torch_action)

            if env.sim.is_stopped(): break
            if args.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args.num_demos:
                print(f"✔ Recorded {args.num_demos} demonstrations.")
                break

            rate.sleep(env)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
