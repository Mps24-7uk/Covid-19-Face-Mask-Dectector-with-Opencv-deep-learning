
import numpy as np
import carb.input



class CustomTeleopController15DOF:
    def __init__(self, pos_sensitivity=0.05, rot_sensitivity=0.1, base_move=0.1):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._input.get_keyboard()

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_move = base_move

        self.gripper_state = {"left": 0.0, "right": 0.0}
        self.toggle_cache = {"6": False, "7": False}

    def is_pressed(self, key_name):
        key_enum = getattr(carb.input.KeyboardInput, key_name.upper(), None)
        return self._keyboard.is_key_down(key_enum) if key_enum else False

    def advance(self):
        action = np.zeros(15)

        # Left Arm
        if self.is_pressed("Q"): action[0] += self.pos_sensitivity
        if self.is_pressed("A"): action[0] -= self.pos_sensitivity
        if self.is_pressed("W"): action[1] += self.pos_sensitivity
        if self.is_pressed("S"): action[1] -= self.pos_sensitivity
        if self.is_pressed("E"): action[2] += self.pos_sensitivity
        if self.is_pressed("D"): action[2] -= self.pos_sensitivity
        if self.is_pressed("Z"): action[3] += self.rot_sensitivity
        if self.is_pressed("X"): action[3] -= self.rot_sensitivity
        if self.is_pressed("T"): action[4] += self.rot_sensitivity
        if self.is_pressed("G"): action[4] -= self.rot_sensitivity
        if self.is_pressed("R"): action[5] += self.rot_sensitivity
        if self.is_pressed("F"): action[5] -= self.rot_sensitivity

        # Right Arm
        if self.is_pressed("Y"): action[6] += self.pos_sensitivity
        if self.is_pressed("H"): action[6] -= self.pos_sensitivity
        if self.is_pressed("U"): action[7] += self.pos_sensitivity
        if self.is_pressed("J"): action[7] -= self.pos_sensitivity
        if self.is_pressed("I"): action[8] += self.pos_sensitivity
        if self.is_pressed("K"): action[8] -= self.pos_sensitivity
        if self.is_pressed("V"): action[9]  += self.rot_sensitivity
        if self.is_pressed("B"): action[9]  -= self.rot_sensitivity
        if self.is_pressed("N"): action[10] += self.rot_sensitivity
        if self.is_pressed("M"): action[10] -= self.rot_sensitivity
        if self.is_pressed("P"): action[11] += self.rot_sensitivity
        if self.is_pressed("L"): action[11] -= self.rot_sensitivity

        # Left Gripper toggle on "6"
        if self.is_pressed("6") and not self.toggle_cache["6"]:
            self.gripper_state["left"] = 1.0 - self.gripper_state["left"]
            self.toggle_cache["6"] = True
        elif not self.is_pressed("6"):
            self.toggle_cache["6"] = False

        # Right Gripper toggle on "7"
        if self.is_pressed("7") and not self.toggle_cache["7"]:
            self.gripper_state["right"] = 1.0 - self.gripper_state["right"]
            self.toggle_cache["7"] = True
        elif not self.is_pressed("7"):
            self.toggle_cache["7"] = False

        action[12] = self.gripper_state["left"]
        action[13] = self.gripper_state["right"]

        # Base Movement
        if self.is_pressed("UP"):
            action[14] = self.base_move
        elif self.is_pressed("DOWN"):
            action[14] = -self.base_move

        return action
