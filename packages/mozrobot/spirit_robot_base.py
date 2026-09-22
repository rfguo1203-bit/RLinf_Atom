from typing import Dict, Tuple, Any
import logging
import numpy as np
import torch


from mozrobot.envs import image_tools


class SpiritRobotBase:
    """
    Base class for Spirit robots, providing common feature definitions.
    The features' definition are according to the document's convention:
        document: https://nwd4iy9rd2s.feishu.cn/wiki/KvpMwspYZiJDeQkwXqeczbVKnpc
        version: v1
    """
    # 机械单元名称
    SPIRIT_MU_NAMES = ["leftarm", "rightarm", "torso", "neck", "base"]
    # 末端执行器名称
    SPIRIT_END_EFFECTOR_NAMES = ["gripper", "hand"]
    # 机械单元观测维度
    SPIRIT_MU_OBS_DIMENSIONS = ["cmd", "state"]
    # 机械单元运动空间
    SPIRIT_MU_MOTION_SPACES = ["cart", "joint"]
    # 机械单元数据字段
    # SPIRIT_MU_DATA_FIELDS = ["pos", "vel", "trq"]
    # Only pos is used in the current version
    SPIRIT_MU_DATA_FIELDS = ["pos"]
    # 末端执行器数据字段
    # Only pos is used in the current version
    # SPIRIT_END_EFFECTOR_DATA_FIELDS = ["pos", "vel", "force"]
    SPIRIT_END_EFFECTOR_DATA_FIELDS = ["pos"]
    # 轮子数据字段
    SPIRIT_BASE_DATA_FIELDS = ["speed"]

    def __init__(
        self,
        config: Any,
        enabled_mechunits: dict[str, int],
        enabled_end_effectors: dict[str, tuple[str, int]] = {"leftarm": ("gripper", 1), "rightarm": ("gripper", 1)},
    ) -> None:
        """
        Initializes the RobotBase with a configuration object.

        Args:
            config: A configuration object specific to the robot.
            enabled_mechunits: A dictionary of enabled mechunits and their DOFs.
                                The key is the mechunit name, the value is the DOF.
            enabled_end_effectors: A dictionary of enabled end-effectors.
                                The key is the arm name, the value is a tuple of end-effector name and DOF.
                                Order: left arm end-effector, right arm end-effector.
        """
        self.config = config
        self._teleop_device_enabled = False

        arm_cnt = 0
        for mechunit_name in enabled_mechunits:
            if mechunit_name not in SpiritRobotBase.SPIRIT_MU_NAMES:
                raise ValueError(f"Invalid mechunit name: {mechunit_name}")
            if "arm" in mechunit_name.split("_")[0]:
                arm_cnt += 1

        for mu_name, ee_conf in enabled_end_effectors.items():
            if mu_name not in SpiritRobotBase.SPIRIT_MU_NAMES or "arm" not in mu_name.split("_")[0]:
                raise ValueError(f"Invalid end-effector's mechunit name: {mu_name}, only arm mechunit is supported")
            if ee_conf[0] not in SpiritRobotBase.SPIRIT_END_EFFECTOR_NAMES:
                raise ValueError(f"Invalid end-effector name: {ee_conf[0]}")

        if arm_cnt != len(enabled_end_effectors):
            raise ValueError(f"The number of enabled arms ({arm_cnt}) does not match the number of enabled end-effectors ({len(enabled_end_effectors)})")

        self.enabled_mechunits = enabled_mechunits
        self.enabled_end_effectors = enabled_end_effectors
        self.cached_motor_features = None

    @staticmethod
    def _get_single_mechunit_features(mechunit_name: str, DOF: int) -> Dict[str, Dict[str, Any]]:
        """
        Defines the features for end-effector of the robot single arm.
        """
        features = {}
        for obs_dimension in SpiritRobotBase.SPIRIT_MU_OBS_DIMENSIONS:
            for motion_space in SpiritRobotBase.SPIRIT_MU_MOTION_SPACES:
                for field_name in SpiritRobotBase.SPIRIT_MU_DATA_FIELDS:
                    key = f"{mechunit_name}_{obs_dimension}_{motion_space}_{field_name}"
                    data_size = 6 if motion_space == "cart" else DOF

                    if motion_space == "cart" and field_name == "trq":
                        names = ["fx", "fy", "fz", "tx", "ty", "tz"]
                    else:
                        names = ["x", "y", "z", "rx", "ry", "rz"] if motion_space == "cart" else [f"joint{i}" for i in range(DOF)]
                    features[key] = {
                        "dtype": "float32",
                        "shape": (data_size,),
                        "names": names
                    }
        return features

    @staticmethod
    def _get_end_effector_features(arm_name: str, end_effector_name: str, DOF: int) -> Dict[str, Dict[str, Any]]:
        """
        Defines the features for end-effector of the robot single arm.
        """
        features = {}
        for obs_dimension in SpiritRobotBase.SPIRIT_MU_OBS_DIMENSIONS:
            for field_name in SpiritRobotBase.SPIRIT_END_EFFECTOR_DATA_FIELDS:
                key = f"{arm_name}_{end_effector_name}_{obs_dimension}_{field_name}"
                features[key] = {
                    "dtype": "float32",
                    "shape": (DOF,),
                    "names": [f"{i}" for i in range(DOF)],
                }
        return features

    @staticmethod
    def _get_single_arm_mechunit_features(mechunit_name: str, DOF: int, enabled_end_effector: Tuple[str, int]) -> Dict[str, Dict[str, Any]]:
        """
        Defines the features for a single arm of the robot.
        """
        assert "arm" in mechunit_name.split("_")[0]

        # get mechunit features
        features = SpiritRobotBase._get_single_mechunit_features(mechunit_name, DOF)
        # if robot has more than 6 DOF, add psi features
        if DOF > 6:
            # get arm psi features
            features.update(
                {
                    f"{mechunit_name}_{obs_dimension}_psi": {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["psi"],
                    }
                    for obs_dimension in SpiritRobotBase.SPIRIT_MU_OBS_DIMENSIONS
                }
            )

        # update gripper features
        end_effector_name, end_effector_DOF = enabled_end_effector
        features.update(SpiritRobotBase._get_end_effector_features(mechunit_name, end_effector_name, end_effector_DOF))

        return features

    @staticmethod
    def _get_base_features(mechunit_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Defines the features for the base of the robot.
        """
        features = {}
        for obs_dimension in SpiritRobotBase.SPIRIT_MU_OBS_DIMENSIONS:
            for field_name in SpiritRobotBase.SPIRIT_BASE_DATA_FIELDS:
                key = f"{mechunit_name}_{obs_dimension}_{field_name}"
                features[key] = {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": [f"{field_name}"],
                }
        return features

    def _get_all_robot_features(self) -> Dict[str, Dict[str, Any]]:
        features = {}
        for mechunit_name, DOF in self.enabled_mechunits.items():
            if mechunit_name == "base":
                features.update(SpiritRobotBase._get_base_features(mechunit_name))
            elif "arm" in mechunit_name.split("_")[0]:
                features.update(SpiritRobotBase._get_single_arm_mechunit_features(mechunit_name, DOF, self.enabled_end_effectors[mechunit_name]))
            else:
                features.update(SpiritRobotBase._get_single_mechunit_features(mechunit_name, DOF))
        return features

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.config.cameras.items():
            cam_ft[cam_key] = {
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        if self.cached_motor_features is None:
            self.cached_motor_features = self._get_all_robot_features()
        return self.cached_motor_features

    def send_action(self, action, action_time: float = None) -> torch.Tensor:
        self._env.send_action(action, action_time)
        return action
    
    def send_actions(self, actions):
        self._env.send_actions(actions)
        return actions

    def capture_observation(self) -> dict:
        obs_dict = self.capture_robot_observation()
        obs_dict.update(self.capture_images())

        return obs_dict

    def capture_images(self) -> dict:
        obs = self._env.get_images_observation()
        obs_dict = {}
        for cam_name, image in obs["images"].items():
            if "_depth" in cam_name:
                continue
            obs_dict[cam_name] = np.ascontiguousarray(
                image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        image,
                        self.config.cameras[cam_name].height,
                        self.config.cameras[cam_name].width,
                    )
                )[:, :, ::-1]
            )
        return obs_dict

    def capture_robot_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("Robot not connected.")

        obs = self._env.get_robot_observation()

        obs_dict = {}

        left_cart_pose, right_cart_pose = obs["cartpos"]
        left_arm, left_grip, right_arm, right_grip = obs["qpos"]
        if "psi" in obs:
            left_psi, right_psi = obs["psi"]
            obs_dict.update({
                f'leftarm_state_psi': np.asarray([left_psi], dtype=np.float32),
                f'rightarm_state_psi': np.asarray([right_psi], dtype=np.float32),
            })
        if "torso_cartpos" in obs and "torso_qpos" in obs:
            torso_cart_pose = obs["torso_cartpos"]
            torso_qpos = obs["torso_qpos"]
            obs_dict.update({
                f'torso_state_cart_pos': np.asarray(torso_cart_pose, dtype=np.float32),
                f'torso_state_joint_pos': np.asarray(torso_qpos, dtype=np.float32),
            })
        if "base_speed" in obs:
            obs_dict.update({
                f'base_state_speed': np.asarray(obs["base_speed"], dtype=np.float32),
            })

        obs_dict.update({
            f'leftarm_state_cart_pos': np.asarray(left_cart_pose, dtype=np.float32),
            f'leftarm_state_joint_pos': np.asarray(left_arm, dtype=np.float32),
            f'rightarm_state_cart_pos': np.asarray(right_cart_pose, dtype=np.float32),
            f'rightarm_state_joint_pos': np.asarray(right_arm, dtype=np.float32),
            f'leftarm_gripper_state_pos': np.asarray(left_grip, dtype=np.float32),
            f'rightarm_gripper_state_pos': np.asarray(right_grip, dtype=np.float32),
        })

        return obs_dict

    def support_soft_real_time(self) -> bool:
        return False

    def completing_dataset_frame(self, frame: dict, fill_empty_cmd_with_state: bool = True, padding_empty_features: bool = False) -> dict:
        """
        Completing the dataset frame with missed features.
        Pad the missed features with zeros.
        Args:
            frame: The dataset frame to be completed.
            fill_empty_cmd_with_state: Whether to fill the empty cmd with state.
            padding_empty_features: Whether to pad the empty features with zeros.
        """

        missed_features = [k for k in self.motor_features.keys() if k not in frame]
        for missed_key in missed_features:
            feature = self.motor_features[missed_key]

            # For missed cmd fields, try to use corresponding state data
            if fill_empty_cmd_with_state and "_cmd_" in missed_key:
                state_key = missed_key.replace("_cmd_", "_state_")
                if state_key in frame:
                    frame[missed_key] = frame[state_key]
                    continue

            # For missed state fields, try to use corresponding cmd data
            if fill_empty_cmd_with_state and "_state_" in missed_key:
                cmd_key = missed_key.replace("_state_", "_cmd_")
                if cmd_key in frame:
                    frame[missed_key] = frame[cmd_key]
                    continue

            if padding_empty_features:
                frame[missed_key] = np.zeros(shape=feature["shape"], dtype=feature["dtype"])

        return frame

    @property
    def teleop_device_enabled(self) -> bool:
        return self._teleop_device_enabled

    @property
    def camera_resolution(self) -> dict:
        return {cam_name: (cam["shape"][1], cam["shape"][0]) for cam_name, cam in self.camera_features.items()}
