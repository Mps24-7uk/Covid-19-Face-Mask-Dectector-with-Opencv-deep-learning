import numpy as np
import weakref
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase


class CustomTeleopController15DOF(DeviceBase):
    """Gamepad controller for 15-DOF robot control:
    - Left Arm: 6 DOF
    - Right Arm: 6 DOF
    - Left + Right Gripper toggle: 2 DOF
    - Base movement (forward/backward): 1 DOF
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.6, base_sensitivity=1.0, dead_zone=0.01):
        carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_sensitivity = base_sensitivity
        self.dead_zone = dead_zone

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)

        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )

        self._create_key_bindings()
        self._gripper_state = {"left": False, "right": False}
        self._base_move = 0.0
        self._delta_pose_raw = {
            "left": np.zeros((2, 6)),
            "right": np.zeros((2, 6))
        }

    def __del__(self):
        self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)

    def reset(self):
        self._gripper_state = {"left": False, "right": False}
        self._base_move = 0.0
        for side in self._delta_pose_raw:
            self._delta_pose_raw[side].fill(0.0)

    def advance(self) -> np.ndarray:
        left_pos = self._resolve_command_buffer(self._delta_pose_raw["left"][:, :3])
        left_rot = self._resolve_command_buffer(self._delta_pose_raw["left"][:, 3:])
        right_pos = self._resolve_command_buffer(self._delta_pose_raw["right"][:, :3])
        right_rot = self._resolve_command_buffer(self._delta_pose_raw["right"][:, 3:])

        left_rotvec = Rotation.from_euler("XYZ", left_rot).as_rotvec()
        right_rotvec = Rotation.from_euler("XYZ", right_rot).as_rotvec()

        action = np.concatenate([
            left_pos, left_rotvec,
            right_pos, right_rotvec,
            [float(self._gripper_state["left"])],
            [float(self._gripper_state["right"])],
            [self._base_move],
        ])

        self._base_move = 0.0  # reset per frame
        return action

    def _on_gamepad_event(self, event, *args, **kwargs):
        val = event.value
        if abs(val) < self.dead_zone:
            val = 0

        # Gripper toggle: X = left, Y = right
        if event.input == carb.input.GamepadInput.X and val > 0.5:
            self._gripper_state["left"] = not self._gripper_state["left"]
        if event.input == carb.input.GamepadInput.Y and val > 0.5:
            self._gripper_state["right"] = not self._gripper_state["right"]

        # Base movement
        if event.input == carb.input.GamepadInput.LEFT_SHOULDER:
            self._base_move = -self.base_sensitivity
        elif event.input == carb.input.GamepadInput.RIGHT_SHOULDER:
            self._base_move = +self.base_sensitivity

        # Pose mappings
        for side in ["left", "right"]:
            if event.input in self._INPUT_BINDINGS[side]:
                direction, axis, gain = self._INPUT_BINDINGS[side][event.input]
                self._delta_pose_raw[side][direction, axis] = gain * val

        return True

    def _create_key_bindings(self):
        # axis index: 0=x, 1=y, 2=z, 3=roll, 4=pitch, 5=yaw
        self._INPUT_BINDINGS = {
            "left": {
                carb.input.GamepadInput.LEFT_STICK_UP: (0, 0, self.pos_sensitivity),
                carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0, self.pos_sensitivity),
                carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 1, self.pos_sensitivity),
                carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 1, self.pos_sensitivity),
                carb.input.GamepadInput.RIGHT_STICK_UP: (0, 2, self.pos_sensitivity),
                carb.input.GamepadInput.RIGHT_STICK_DOWN: (1, 2, self.pos_sensitivity),
                carb.input.GamepadInput.DPAD_LEFT: (1, 3, self.rot_sensitivity),
                carb.input.GamepadInput.DPAD_RIGHT: (0, 3, self.rot_sensitivity),
                carb.input.GamepadInput.DPAD_DOWN: (0, 4, self.rot_sensitivity),
                carb.input.GamepadInput.DPAD_UP: (1, 4, self.rot_sensitivity),
                carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 5, self.rot_sensitivity),
                carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 5, self.rot_sensitivity),
            },
            "right": {
                carb.input.GamepadInput.A: (0, 0, self.pos_sensitivity),
                carb.input.GamepadInput.B: (1, 0, self.pos_sensitivity),
                carb.input.GamepadInput.LEFT_THUMB: (1, 1, self.pos_sensitivity),
                carb.input.GamepadInput.RIGHT_THUMB: (0, 1, self.pos_sensitivity),
                carb.input.GamepadInput.BACK: (0, 2, self.pos_sensitivity),
                carb.input.GamepadInput.START: (1, 2, self.pos_sensitivity),
                # Extend with custom buttons for right-arm rotation if needed
            },
        }

    def _resolve_command_buffer(self, raw_command: np.ndarray) -> np.ndarray:
        signs = raw_command[1] > raw_command[0]
        delta = raw_command.max(axis=0)
        delta[signs] *= -1
        return delta
