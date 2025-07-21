
# SPDX-License-Identifier: BSD-3-Clause
# Updated record_demos.py for CustomTeleopController15DOF

import argparse
import contextlib
import os
import time
import torch
import gymnasium as gym
import numpy as np
import omni.log
import omni.ui as ui

from isaaclab.app import AppLauncher
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.envs.ui import EmptyWindow

from custom_teleop_controller_15dof import CustomTeleopController15DOF
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--teleop_device", type=str, default="custom15dof")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5")
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--num_demos", type=int, default=0)
parser.add_argument("--num_success_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

# --- Launch App ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa
import isaaclab_mimic.envs  # noqa

class RateLimiter:
    def __init__(self, hz): self.hz = hz; self.sleep_duration = 1.0 / hz; self.last_time = time.time(); self.render_period = min(0.033, self.sleep_duration)
    def sleep(self, env):
        next_time = self.last_time + self.sleep_duration
        while time.time() < next_time: time.sleep(self.render_period); env.sim.render()
        self.last_time = next_time
        while self.last_time < time.time(): self.last_time += self.sleep_duration

def pre_process_actions(teleop_data, num_envs, device):
    return torch.tensor(teleop_data, dtype=torch.float32, device=device).unsqueeze(0)

def main():
    rate_limiter = RateLimiter(args_cli.step_hz)

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn("No success condition detected.")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    teleop = CustomTeleopController15DOF(pos_sensitivity=0.2, rot_sensitivity=0.5, base_sensitivity=0.1)
    env.sim.reset(); env.reset(); teleop.reset()

    current_demos = 0; success_count = 0; should_reset = False
    label_text = f"Recorded {current_demos} successful demonstrations."

    instruction_display = InstructionDisplay(args_cli.teleop_device)
    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
        instruction_display.set_labels(subtask_label, demo_label)

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            action = teleop.advance()
            action_tensor = pre_process_actions(action, env.num_envs, env.device)
            obv = env.step(action_tensor)

            if success_term and bool(success_term.func(env, **success_term.params)[0]):
                success_count += 1
                if success_count >= args_cli.num_success_steps:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
                    env.recorder_manager.export_episodes([0])
                    should_reset = True
            else:
                success_count = 0

            if env.recorder_manager.exported_successful_episode_count > current_demos:
                current_demos = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_demos} successful demonstrations."
                print(label_text)

            if should_reset:
                env.sim.reset(); env.recorder_manager.reset(); env.reset()
                success_count = 0; should_reset = False
                instruction_display.show_demo(label_text)

            if args_cli.num_demos > 0 and current_demos >= args_cli.num_demos:
                print(f"Finished {args_cli.num_demos} demos.")
                break

            if env.sim.is_stopped(): break
            rate_limiter.sleep(env)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
