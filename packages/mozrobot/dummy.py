from typing import List, Optional, Dict, Any  # noqa: UP035
import logging

import torch
import numpy as np
import scipy.spatial.transform as st


from mozrobot.configs import DummyRobotConfig
from mozrobot.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from mozrobot.spirit_robot_base import SpiritRobotBase


class DummyEnv:
    def __init__(self):
        pass
    
    def get_robot_observation(self) -> Dict[str, Any]:
        obs = {
            "cartpos": (np.zeros((6,), dtype=np.float32), np.zeros((6,), dtype=np.float32)),
            "qpos": (np.zeros((6,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((6,), dtype=np.float32), np.zeros((1,), dtype=np.float32)),
            "psi": (np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)),
            "torso_cartpos": np.zeros((6,), dtype=np.float32),
            "torso_qpos": np.zeros((6,), dtype=np.float32),
        }
        return obs
    
    def get_images_observation(self) -> Dict[str, Any]:
        images = {
            "cam_high": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        return {"images": images}
    
    def send_action(self, action: Dict[str, np.ndarray]) -> None:
        pass


class DummyRobot(SpiritRobotBase):
    def __init__(
        self,
        config: DummyRobotConfig,
    ) -> None:
        super().__init__(config, enabled_mechunits={"leftarm": 6, "rightarm": 6}, enabled_end_effectors={"leftarm": ("gripper", 1), "rightarm": ("gripper", 1)})
        self.robot_type = self.config.type
        self.is_connected = False
        self.cameras = [None]*3

    def connect(self) -> None:
        self._env = DummyEnv()
        self.is_connected = True

    def go_back_home(self) -> None:
        print("DummyRobot: Going back home.")

    def reset_teleop_zero_pos(self, relative=False):
        print("DummyRobot: Resetting teleop zero position.")
    
    def teleop_step(
        self, record_data=False, record_image=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        
        action = {
            f'leftarm_gripper_cmd_pos': np.zeros((1,), dtype=np.float32),
            f'leftarm_cmd_cart_pos': np.zeros((6,), dtype=np.float32),
            f'leftarm_cmd_joint_pos': np.zeros((6,), dtype=np.float32),
            f'rightarm_gripper_cmd_pos': np.zeros((1,), dtype=np.float32),
            f'rightarm_cmd_cart_pos': np.zeros((6,), dtype=np.float32),
            f'rightarm_cmd_joint_pos': np.zeros((6,), dtype=np.float32),
        }

        observation = self.capture_robot_observation()
        if record_image:
            images = self.capture_images()
            observation.update(images)

        return observation, action

    def disconnect(self) -> None:
        self._env = None
        self.is_connected = False
