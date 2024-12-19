from typing import Dict, Any, List
import numpy as np

OASIS_ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


# Define action functions
def forward_backward_action(action: int) -> int:
    """
    Forward and Backward Action

    Description:
        Controls forward and backward movement.

    Input Arguments:
        action (int):
            0 - noop
            1 - forward
            2 - back

    Returns:
        int: Validated action value.

    Raises:
        ValueError: If action is not in [0, 1, 2].
    """
    if action not in [0, 1, 2]:
        raise ValueError(
            "ForwardBackwardAction must be 0 (noop), 1 (forward), or 2 (back)."
        )
    return action


def move_left_right_action(action: int) -> int:
    """
    Move Left and Right Action

    Description:
        Controls left and right movement.

    Input Arguments:
        action (int):
            0 - noop
            1 - move left
            2 - move right

    Returns:
        int: Validated action value.

    Raises:
        ValueError: If action is not in [0, 1, 2].
    """
    if action not in [0, 1, 2]:
        raise ValueError(
            "MoveLeftRightAction must be 0 (noop), 1 (move left), or 2 (move right)."
        )
    return action


def jump_sneak_sprint_action(action: int) -> int:
    """
    Jump, Sneak, and Sprint Action

    Description:
        Controls jumping, sneaking, and sprinting.

    Input Arguments:
        action (int):
            0 - noop
            1 - jump
            2 - sneak
            3 - sprint

    Returns:
        int: Validated action value.

    Raises:
        ValueError: If action is not in [0, 1, 2, 3].
    """
    if action not in [0, 1, 2, 3]:
        raise ValueError(
            "JumpSneakSprintAction must be 0 (noop), 1 (jump), 2 (sneak), or 3 (sprint)."
        )
    return action


def camera_delta_pitch_action(camera_x: int) -> int:
    """
    Camera Delta Pitch Action

    Description:
        Controls the camera's pitch (vertical movement).

    Input Arguments:
        camera_x (int):
            Range: 0 to 24 corresponding to -180° to 180°.

    Returns:
        int: Mapped and validated camera pitch value.

    Raises:
        ValueError: If camera_x is not in [0, 24].
    """
    if not isinstance(camera_x, int):
        raise TypeError("CameraDeltaPitchAction requires an integer input.")
    if not 0 <= camera_x <= 24:
        raise ValueError("CameraDeltaPitchAction must be in the range [0, 24].")
    return camera_x


def camera_delta_yaw_action(camera_y: int) -> int:
    """
    Camera Delta Yaw Action

    Description:
        Controls the camera's yaw (horizontal movement).

    Input Arguments:
        camera_y (int):
            Range: 0 to 24 corresponding to -180° to 180°.

    Returns:
        int: Mapped and validated camera yaw value.

    Raises:
        ValueError: If camera_y is not in [0, 24].
    """
    if not isinstance(camera_y, int):
        raise TypeError("CameraDeltaYawAction requires an integer input.")
    if not 0 <= camera_y <= 24:
        raise ValueError("CameraDeltaYawAction must be in the range [0, 24].")
    return camera_y


def functional_action(action: int) -> int:
    """
    Functional Actions

    Description:
        Controls functional actions like use, drop, attack, etc.

    Input Arguments:
        action (int):
            0 - noop
            1 - use
            2 - drop
            3 - attack
            4 - craft
            5 - equip
            6 - place
            7 - destroy

    Returns:
        int: Validated action value.

    Raises:
        ValueError: If action is not in [0, 1, 2, 3, 4, 5, 6, 7].
    """
    if action not in range(8):
        raise ValueError("FunctionalAction must be between 0 (noop) and 7 (destroy).")
    return action


def craft_argument_action(item_id: int) -> int:
    """
    Argument for Craft Action.
    Only effective for when the functional_action is 4 - craft.

    Description:
        Specifies the item to be crafted.

    Input Arguments:
        item_id (int):
            Range: 0 to 243 representing different craftable items.

    Returns:
        int: Validated item_id.

    Raises:
        ValueError: If item_id is not in [0, 243].
    """
    if not isinstance(item_id, int):
        raise TypeError("CraftArgumentAction requires an integer input.")
    if not 0 <= item_id <= 243:
        raise ValueError("CraftArgumentAction must be in the range [0, 243].")
    return item_id


# The item selection range in MineDojo is [0, 40], but we limit to hotbar slots [1, 9] to align with OASIS
# EQUIP_SLOTS = {
#     "mainhand": 0,
#     "offhand": 40,
#     "head": 39,
#     "chest": 38,
#     "legs": 37,
#     "feet": 36,
# }
# MIN_SLOT_IDX = 0
# MAX_SLOT_IDX = 40
def equip_place_destroy_argument_action(inventory_slot: int) -> int:
    """
    Item selection as an argument for Equip, Place, and Destroy Actions.
    Only effective for when the functional_action is one of the following:
        5 - equip
        6 - place
        7 - destroy

    Description:
        Specifies the inventory slot index.

    Input Arguments:
        inventory_slot (int):
            Range: 1 to 9 representing different inventory slots.

    Returns:
        int: Validated inventory_slot.

    Raises:
        ValueError: If inventory_slot is not in [1, 9].
    """
    if not isinstance(inventory_slot, int):
        raise TypeError("EquipPlaceDestroyArgumentAction requires an integer input.")
    if not 0 <= inventory_slot <= 9:
        # Keep 0 for noop
        raise ValueError("EquipPlaceDestroyArgumentAction must be in the range [1, 9].")
    return inventory_slot


class ActionInterface:
    def __init__(self, actions: Dict[str, str]):
        """Initialize the ActionInterface with the provided actions.

        Args:
            actions (Dict[str, str]): Dictionary of action inputs.
            Example:
                {
                    "forward_backward_action": "1",
                    "jump_sneak_sprint_action": "1",
                    "camera_delta_pitch_action": "12",
                    "camera_delta_yaw_action": "12",
                    "functional_action": "1",
                    ...
                }

        """
        # Parse and validate action inputs from the JSON-formatted dictionary
        self.actions = actions
        if "forward_backward_action" in actions:
            self.forward_backward = forward_backward_action(
                int(actions["forward_backward_action"])
            )
        else:
            self.forward_backward = forward_backward_action(0)

        if "move_left_right_action" in actions:
            self.move_left_right = move_left_right_action(
                int(actions["move_left_right_action"])
            )
        else:
            self.move_left_right = move_left_right_action(0)

        if "jump_sneak_sprint_action" in actions:
            self.jump_sneak_sprint = jump_sneak_sprint_action(
                int(actions["jump_sneak_sprint_action"])
            )
        else:
            self.jump_sneak_sprint = jump_sneak_sprint_action(0)

        if "camera_delta_pitch_action" in actions:
            self.camera_pitch = camera_delta_pitch_action(
                int(actions["camera_delta_pitch_action"])
            )
        else:
            self.camera_pitch = camera_delta_pitch_action(12)

        if "camera_delta_yaw_action" in actions:
            self.camera_yaw = camera_delta_yaw_action(
                int(actions["camera_delta_yaw_action"])
            )
        else:
            self.camera_yaw = camera_delta_yaw_action(12)

        if "functional_action" in actions:
            self.functional_action = functional_action(
                int(actions["functional_action"])
            )
        else:
            self.functional_action = functional_action(0)

        if "craft_argument_action" in actions:
            self.craft_argument = craft_argument_action(
                int(actions["craft_argument_action"])
            )
        else:
            self.craft_argument = craft_argument_action(0)

        if "equip_place_destroy_argument_action" in actions:
            self.equip_place_destroy_argument = equip_place_destroy_argument_action(
                int(actions["equip_place_destroy_argument_action"])
            )
        else:
            self.equip_place_destroy_argument = equip_place_destroy_argument_action(0)

    @staticmethod
    def get_actions_prompt() -> Dict[str, Any]:
        return {
            "forward_backward_action": forward_backward_action.__doc__,
            "move_left_right_action": move_left_right_action.__doc__,
            "jump_sneak_sprint_action": jump_sneak_sprint_action.__doc__,
            "camera_delta_pitch_action": camera_delta_pitch_action.__doc__,
            "camera_delta_yaw_action": camera_delta_yaw_action.__doc__,
            "functional_action": functional_action.__doc__,
            "craft_argument_action": craft_argument_action.__doc__,
            "equip_place_destroy_argument_action": equip_place_destroy_argument_action.__doc__,
        }

    def to_oasis_format(self) -> Dict[str, Any]:
        # Convert to OASIS WorldModel input format
        action_dict = {
            "forward": 1 if self.forward_backward == 1 else 0,
            "back": 1 if self.forward_backward == 2 else 0,
            "left": 1 if self.move_left_right == 1 else 0,
            "right": 1 if self.move_left_right == 2 else 0,
            "camera": [
                compress_mouse(self.camera_pitch),  # Converted using compress_mouse
                compress_mouse(self.camera_yaw),  # Converted using compress_mouse
            ],
            "jump": 1 if self.jump_sneak_sprint == 1 else 0,
            "sneak": 1 if self.jump_sneak_sprint == 2 else 0,
            "sprint": 1 if self.jump_sneak_sprint == 3 else 0,
            "swapHands": 0,
            "attack": 1 if self.functional_action == 3 else 0,
            "use": 1 if self.functional_action == 1 else 0,
            "pickItem": 0,
            "drop": 1 if self.functional_action == 2 else 0,
            "craft": 1 if self.functional_action == 4 else 0,
            "equip": 1 if self.functional_action == 5 else 0,
            "place": 1 if self.functional_action == 6 else 0,
            "destroy": 1 if self.functional_action == 7 else 0,
            "inventory": 0,
            "ESC": self.actions.get("ESC", 0),
            "hotbar.1": 1 if self.equip_place_destroy_argument == 1 else 0,
            "hotbar.2": 1 if self.equip_place_destroy_argument == 2 else 0,
            "hotbar.3": 1 if self.equip_place_destroy_argument == 3 else 0,
            "hotbar.4": 1 if self.equip_place_destroy_argument == 4 else 0,
            "hotbar.5": 1 if self.equip_place_destroy_argument == 5 else 0,
            "hotbar.6": 1 if self.equip_place_destroy_argument == 6 else 0,
            "hotbar.7": 1 if self.equip_place_destroy_argument == 7 else 0,
            "hotbar.8": 1 if self.equip_place_destroy_argument == 8 else 0,
            "hotbar.9": 1 if self.equip_place_destroy_argument == 9 else 0,
        }
        return action_dict

    def to_minedojo_format(self) -> List[int]:
        """
        Convert to MineDojo MultiDiscrete action format.
        Action Space:
        MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
        """
        action = np.zeros(8, dtype=int)  # Initialize all action dimensions to noop

        # Index 0: Forward and backward
        if self.forward_backward == 1:
            action[0] = 1  # forward
        elif self.forward_backward == 2:
            action[0] = 2  # back

        # Index 1: Move left and right
        if self.move_left_right == 1:
            action[1] = 1  # move left
        elif self.move_left_right == 2:
            action[1] = 2  # move right

        # Index 2: Jump, sneak, and sprint
        if self.jump_sneak_sprint == 1:
            action[2] = 1  # jump
        elif self.jump_sneak_sprint == 2:
            action[2] = 2  # sneak
        elif self.jump_sneak_sprint == 3:
            action[2] = 3  # sprint

        # Index 3: Camera delta pitch (discretized into 15-degree intervals)
        action[3] = self.camera_pitch

        # Index 4: Camera delta yaw (discretized into 15-degree intervals)
        action[4] = self.camera_yaw

        # Index 5: Functional actions
        action[5] = self.functional_action  # 0: noop, 1: use, etc.

        # Index 6: Argument for "craft" (244 possible items)
        if self.functional_action == 4:
            action[6] = self.craft_argument
        else:
            action[6] = 0  # noop

        # Index 7: Argument for "equip", "place", and "destroy" (36 inventory slots)
        if self.functional_action in [5, 6, 7]:
            action[7] = self.equip_place_destroy_argument
        else:
            action[7] = 0  # noop

        return action.tolist()


def compress_mouse(camera_val: int) -> int:
    """
    Compresses and converts camera action input from MineDojo (0-24) to OASIS domain.

    Args:
        camera_val (int): Camera action value from MineDojo (0 to 24).

    Returns:
        int: Compressed camera action value for OASIS.

    # WARN: Adapt from issue https://github.com/etched-ai/open-oasis/issues/9, not 100% certain
    """
    max_val = 20
    bin_size = 0.5
    mu = 2.7

    # Map 0-24 to -180 to 180 degrees
    degrees = (camera_val - 12) * (360 / 24)

    # Clamp the value
    degrees = np.clip(degrees, -180, 180)

    # Normalize to [-1, 1]
    dx_normalized = degrees / 180.0

    # Non-linear encoding
    v_encode = np.sign(dx_normalized) * (
        np.log(1.0 + mu * np.abs(dx_normalized)) / np.log(1.0 + mu)
    )
    v_encode *= max_val

    # Discretize to bin_size
    return int(round((v_encode + max_val) / bin_size))
