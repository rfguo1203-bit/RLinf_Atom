# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import logging
from typing import Optional

import numpy as np

from mozrobot.envs import robot_utils
from mozrobot.envs.robot_utils import CameraError
from mozrobot.envs.moz1_controller.moz1_arm_driver import Moz1BrigeInterfaceStub

MOZ1_STRUCTURE_MAPPING = {
    "dualarm": (None, 0, 1, None),
    "wholebody_without_base": (0, 1, 2, None),
    "wholebody": (1, 2, 3, 0),
}
class RealEnv():
    """
    Environment for real robot bi-manual manipulation

    Control spaces available:
    1. joints:     Direct joint position control
    2. eef_vec:    End-effector cartesian control with rotation vector
    3. eef_rot6d:  End-effector cartesian control with 6D rotation representation

    Action space (depending on control space):
    - joints mode:
        [left_arm_qpos (7),             # absolute joint position
         left_gripper_positions (1),    # real gripper position (0: close, 0.12: open)
         right_arm_qpos (7),            # absolute joint position
         right_gripper_positions (1),]  # real gripper position (0: close, 0.12: open)

    - eef_vec mode:
        [left_cart_pos (6),             # cartesian pose (px, py, pz, rotvec0, rotvec1, rotvec2), unit: m, rad
         left_gripper_positions (1),    # real gripper position (0: close, 0.12: open)
         right_cart_pos (6),            # cartesian pose (px, py, pz, rotvec0, rotvec1, rotvec2), unit: m, rad
         right_gripper_positions (1),]  # real gripper position (0: close, 0.12: open)

    - eef_rot6d mode:
        [left_cart_pos (9),             # cartesian pose (px, py, pz, rot6d[0:6]), unit: m, -
         left_gripper_positions (1),    # real gripper position (0: close, 0.12: open)
         right_cart_pos (9),            # cartesian pose (px, py, pz, rot6d[0:6]), unit: m, -
         right_gripper_positions (1),]  # real gripper position (0: close, 0.12: open)

    Observation space:
        {"cartpos": Concat[  left_pose_base (6),   # left arm cartesian pose(px, py, pz, rotvec0, rotvec1, rotvec2), unit: m, rad
                            right_pose_base (6)]    # right arm cartesian pose(px, py, pz, rotvec0, rotvec1, rotvec2) unit: m, rad
         "qpos": Concat[    left_arm_qpos (7),          # absolute joint position
                           left_gripper_position (1),   # real gripper position (0: close, 0.12: open)
                           right_arm_qpos (7),          # absolute joint position
                           right_gripper_qpos (1)]      # real gripper position (0: close, 0.12: open)
         "qvel": Concat[    left_arm_qvel (7),          # absolute joint velocity (rad)
                           left_gripper_velocity (1),   # real gripper velocity (pos: opening, neg: closing)
                           right_arm_qvel (7),          # absolute joint velocity (rad)
                           right_gripper_qvel (1)]      # real gripper velocity (pos: opening, neg: closing)
         "images": {        "cam_high": (224x224x3),        # h, w, c, dtype='uint8'
                           "cam_left_wrist": (224x224x3),   # h, w, c, dtype='uint8'
                           "cam_right_wrist": (224x224x3)}  # h, w, c, dtype='uint8'
    """
    def __init__(self, **kwargs):
        # 用户可见的参数
        init_params = {
            'realsense_serials': kwargs.get('realsense_serials', "2-1,2-3,1-4"),
            'camera_resolutions': kwargs.get('camera_resolutions', {"cam_high": (320, 240), "cam_left_wrist": (320, 240), "cam_right_wrist": (320, 240)}),
            'no_camera': kwargs.get('no_camera', False),
            'structure': kwargs.get('structure', "dualarm"),
            'robot_control_hz': kwargs.get('robot_control_hz', 120),
            'simu_mode': kwargs.get('simu_mode', False),
            'virtual_robot': kwargs.get('virtual_robot', False),
            'log_queue': kwargs.get('log_queue', None),
            'enable_logging': kwargs.get('enable_logging', False),
            'enable_soft_realtime': kwargs.get('enable_soft_realtime', False),
            'bind_cpu_idxs': kwargs.get('bind_cpu_idxs', None),
            'disabled_cameras': kwargs.get('disabled_cameras', None),
        }
        logging.debug(f"RealEnv init_params: {init_params}")

        moz1_structure = init_params['structure']
        if moz1_structure not in MOZ1_STRUCTURE_MAPPING:
            raise ValueError(f"Invalid structure: {moz1_structure}. Must be one of: {', '.join(MOZ1_STRUCTURE_MAPPING.keys())}")

        logging.debug(f"RealEnv simu_mode: {init_params['simu_mode']}")

        mu_idxs_list = MOZ1_STRUCTURE_MAPPING[moz1_structure]
        logging.debug(f"moz1 structure : {moz1_structure}, mu_idxs_list: {mu_idxs_list}")

        # 初始化双臂控制器 - Using multiprocessing wrapper for stable frequency
        # Grippers are now handled within the arm_controller
        self.arm_controller = Moz1BrigeInterfaceStub(
            rate_hz=init_params['robot_control_hz'],
            mu_idxs_list=mu_idxs_list,
            timeout=5.0,
            virtual=init_params['virtual_robot'],
            simu_mode=init_params['simu_mode'],
            log_queue=init_params['log_queue'],
            enable_logging=init_params['enable_logging'],
            enable_soft_realtime=init_params['enable_soft_realtime'],
            bind_cpu_idxs=init_params['bind_cpu_idxs']
        )

        self.defalut_joint_positions = {
            "left_arm_joints": None,
            "right_arm_joints": None,
            "torso_joints": None,
            "gripper_positions": None,
        }

        # 初始化相机
        self.image_recorder = robot_utils.ImageRecorder(
            camera_resolutions=init_params['camera_resolutions'],
            realsense_serials=init_params['realsense_serials'],
            simu_mode=init_params['simu_mode'],
            no_camera=init_params['no_camera'],
            disabled_cameras=init_params['disabled_cameras']
        )

        try:
            self.image_recorder.start()
            logging.info("Camera system initialized successfully")
        except CameraError as e:
            logging.error(f"Camera initialization failed: {e}")
            # Clean up any partial camera initialization
            try:
                self.image_recorder.stop(wait=True)
            except Exception as cleanup_error:
                logging.warning(f"Error during camera cleanup: {cleanup_error}")
            raise  # Re-raise to let upper layer handle

        logging.info("system initialized")

    def get_cartpos(self):
        states = self.arm_controller.get_states()
        return states.get('left_arm_ee_pose'), states.get('right_arm_ee_pose')

    def get_robot_arm_psi(self):
        states = self.arm_controller.get_states()
        return states.get('left_arm_psi'), states.get('right_arm_psi')

    def get_qpos(self):
        # 获取关节位置
        states = self.arm_controller.get_states()
        left_arm = states.get('left_arm_joints')
        right_arm = states.get('right_arm_joints')

        # Get gripper positions from unified states
        left_grip_pos = states.get('left_gripper_position', 0.0)
        right_grip_pos = states.get('right_gripper_position', 0.0)

        return left_arm, [left_grip_pos], right_arm, [right_grip_pos]

    def get_torso_cartpos(self):
        states = self.arm_controller.get_states()
        return states.get('torso_ee_pose')

    def get_torso_qpos(self):
        states = self.arm_controller.get_states()
        return states.get('torso_joints')

    def get_base_speed(self):
        states = self.arm_controller.get_states()
        return states.get('base_speed')

    def set_gripper_pose(self, left_pos, right_pos):
        """Set gripper positions using arm_controller unified interface.

        Args:
            left_pos: float, list, or np.ndarray - left gripper position
            right_pos: float, list, or np.ndarray - right gripper position
        """
        # Convert to np.ndarray for consistency with the framework
        left_pos_array = np.asarray([left_pos] if isinstance(left_pos, (int, float)) else left_pos, dtype=np.float32)
        right_pos_array = np.asarray([right_pos] if isinstance(right_pos, (int, float)) else right_pos, dtype=np.float32)

        self.arm_controller.send_action({
            'left_gripper_position': left_pos_array,
            'right_gripper_position': right_pos_array
        })

    def update_reset_position(self, left_arm_joints: Optional[list] = None, right_arm_joints: Optional[list] = None, torso_joints: Optional[list] = None, gripper_positions: Optional[list] = None):
        if left_arm_joints is not None:
            self.defalut_joint_positions["left_arm_joints"] = left_arm_joints
        if right_arm_joints is not None:
            self.defalut_joint_positions["right_arm_joints"] = right_arm_joints
        if torso_joints is not None:
            self.defalut_joint_positions["torso_joints"] = torso_joints
        if gripper_positions is not None:
            self.defalut_joint_positions["gripper_positions"] = gripper_positions

    def move_robot_to_reset_position(self) -> bool:
        if any(value is not None for value in self.defalut_joint_positions.values()):
            logging.info("Moz1 robot move to reset position")
            return self.reset_robot_positions(self.defalut_joint_positions)
        else:
            logging.info("No reset position provided, skipping reset")
            return True

    def reset(self):
        self.move_robot_to_reset_position()

    def enable_external_following_mode(self):
        """
        Enable external following mode.
        If robot is not in external following mode, send_action will not work.
        """
        self.arm_controller.send_control_event("enable_external_following_mode")

    def disable_external_following_mode(self):
        """
        Disable external following mode.
        """
        self.arm_controller.send_control_event("disable_external_following_mode")

    def move_robot_to_workpoint(self, workpoint_name: str):
        """
        Move robot to MovaX specified workpoint.
        Note: Robot must not in external following mode.
        """
        self.arm_controller.send_control_event("move_robot_to_workpoint", workpoint_name)

    def reset_robot_positions(self, robot_positions: dict) -> bool:
        """Reset robot positions.

        Args:
            robot_positions (dict): Robot positions to reset.
                left_arm_joints (list): Left arm joints to reset.
                right_arm_joints (list): Right arm joints to reset.
                torso_joints (list): Torso joints to reset.
                gripper_positions (list): Gripper positions to reset.
        Unit: rad, m

        Returns:
            bool: True if reset successful, False otherwise.
        """
        reset_positions = {}
        if "left_arm_joints" in robot_positions and robot_positions["left_arm_joints"] is not None:
            if len(robot_positions["left_arm_joints"]) != 7:
                logging.error(f"Left arm joints length is not 7: {len(robot_positions['left_arm_joints'])}")
                return False
            reset_positions.update({"left_arm_joints": robot_positions["left_arm_joints"]})

        if "right_arm_joints" in robot_positions and robot_positions["right_arm_joints"] is not None:
            if len(robot_positions["right_arm_joints"]) != 7:
                logging.error(f"Right arm joints length is not 7: {len(robot_positions['right_arm_joints'])}")
                return False
            reset_positions.update({"right_arm_joints": robot_positions["right_arm_joints"]})

        if "torso_joints" in robot_positions and robot_positions["torso_joints"] is not None:
            if len(robot_positions["torso_joints"]) != 6:
                logging.error(f"Torso joints length is not 6: {len(robot_positions['torso_joints'])}")
                return False
            reset_positions.update({"torso_joints": robot_positions["torso_joints"]})

        if "gripper_positions" in robot_positions and robot_positions["gripper_positions"] is not None:
            if len(robot_positions["gripper_positions"]) != 2:
                logging.error(f"Gripper positions length is not 2: {len(robot_positions['gripper_positions'])}")
                return False
            reset_positions.update({"gripper_positions": robot_positions["gripper_positions"]})

        if len(reset_positions) == 0:
            logging.error("No reset positions provided")
            return False

        self.arm_controller.send_control_event("reset_robot_positions", reset_positions)
        return True

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = self.get_robot_observation()
        obs.update(self.get_images_observation())
        return obs

    def get_images_observation(self):
        obs = collections.OrderedDict()
        obs["images"] = self.image_recorder.get_images()
        return obs

    def get_robot_observation(self):
        obs = collections.OrderedDict()
        obs['cartpos'] = self.get_cartpos()
        obs['psi'] = self.get_robot_arm_psi()
        obs["qpos"] = self.get_qpos()

        torso_cartpose = self.get_torso_cartpos()
        if torso_cartpose is not None:
            obs['torso_cartpos'] = torso_cartpose
        torso_qpos = self.get_torso_qpos()
        if torso_qpos is not None:
            obs['torso_qpos'] = torso_qpos

        base_speed = self.get_base_speed()
        if base_speed is not None:
            obs['base_speed'] = base_speed

        return obs

    def get_reward(self):
        return 0

    def send_action(self, action, action_time):
        """
        Send unified action to the robot, must be sent at robot_control_hz frequency.

        Args:
            action (dict): Unified action to the robot.
                gripper_commands:
                    leftarm_gripper_cmd_pos (list):  left arm gripper position.
                        format: [pos]
                        Unit: m
                    rightarm_gripper_cmd_pos (list): Right arm gripper position.
                        format: [pos]
                        Unit: m
                left_arm_commands:
                    You can only send one of the following commands at a time.
                    leftarm_cmd_joint_pos (list): Left arm joints to be executed.
                        format: [j1, j2, j3, j4, j5, j6, j7]
                        Unit: rad
                    leftarm_cmd_cart_pos (list): Left arm cartesian pose to be executed. 
                        format: [x, y, z, rx, ry, rz]
                        Unit: m, rad
                right_arm_commands:
                    You can only send one of the following commands at a time.
                    rightarm_cmd_joint_pos (list): Right arm joints to be executed.
                        format: [j1, j2, j3, j4, j5, j6, j7]
                        Unit: rad
                    rightarm_cmd_cart_pos (list): Right arm cartesian pose to be executed.
                        format: [x, y, z, rx, ry, rz]
                        Unit: m, rad
                torso_commands:
                    You can only send one of the following commands at a time.
                    torso_cmd_joint_pos (list): Torso joints to be executed.
                        format: [j1, j2, j3, j4, j5, j6]
                        Unit: rad
                    torso_cmd_cart_pos (list): Torso cartesian pose to be executed.
                        format: [x, y, z, rx, ry, rz]
                        Unit: m, rad
                base_commands:
                    base_cmd_speed (list): Base speed to be executed.
                        format: [v_x, v_y, v_rot]
                        Unit: m/s, rad/s
            action_time (float): Action time.

        Note:
        1. Robot must in external following mode.
        """
        arm_action = self._prepare_action(action)

        # Send unified action
        if arm_action:
            self.arm_controller.send_action(arm_action, action_time)

    def send_actions(self, actions):
        arm_actions = [(self._prepare_action(action[0]), action[1]) for action in actions]
        # Send unified action
        if arm_actions:
            self.arm_controller.send_actions(arm_actions)

    def _prepare_action(self, action):
        """Build unified action dict for arm_controller.

        All numeric arrays are converted to np.ndarray for consistent processing
        in the underlying driver layer.
        """
        arm_action = {}

        # Handle gripper commands - unified to np.ndarray for consistency
        if "leftarm_gripper_cmd_pos" in action:
            arm_action['left_gripper_position'] = np.asarray(action['leftarm_gripper_cmd_pos'], dtype=np.float32)
        if "rightarm_gripper_cmd_pos" in action:
            arm_action['right_gripper_position'] = np.asarray(action['rightarm_gripper_cmd_pos'], dtype=np.float32)

        # Convert joint commands - unified to np.ndarray
        if 'leftarm_cmd_joint_pos' in action:
            arm_action['left_arm_joints'] = np.asarray(action['leftarm_cmd_joint_pos'], dtype=np.float32)
        if 'rightarm_cmd_joint_pos' in action:
            arm_action['right_arm_joints'] = np.asarray(action['rightarm_cmd_joint_pos'], dtype=np.float32)

        # Convert cartesian commands - unified to np.ndarray
        if 'leftarm_cmd_cart_pos' in action:
            arm_action['left_arm_ee_pose'] = np.asarray(action['leftarm_cmd_cart_pos'], dtype=np.float32)
        if 'rightarm_cmd_cart_pos' in action:
            arm_action['right_arm_ee_pose'] = np.asarray(action['rightarm_cmd_cart_pos'], dtype=np.float32)

        # Handle torso - unified to np.ndarray
        if 'torso_cmd_joint_pos' in action:
            arm_action['torso_joints'] = np.asarray(action['torso_cmd_joint_pos'], dtype=np.float32)
        if 'torso_cmd_cart_pos' in action:
            arm_action['torso_ee_pose'] = np.asarray(action['torso_cmd_cart_pos'], dtype=np.float32)

        # Handle base - unified to np.ndarray
        if 'base_cmd_speed' in action:
            arm_action['base_cmd'] = np.asarray(action['base_cmd_speed'], dtype=np.float32)

        return arm_action

def make_real_env(**kwargs) -> RealEnv:
    """
    创建RealEnv实例的便利函数

    支持的参数：
    - realsense_serials: 相机配置，支持自动识别：
      * USB端口格式: "2-1,2-3,1-4" (包含"-"的格式)
      * 序列号格式: "123456,789012,345678" (纯数字或字母数字)
      * 顺序：上相机，左腕部相机，右腕部相机
    - camera_resolutions: 相机分辨率配置, 使用相机名称作为key, 格式为(width, height)
        (如 "cam_high": (320, 240), "cam_left_wrist": (320, 240), "cam_right_wrist": (320, 240))
      * camera_high: 上相机
      * camera_left_wrist: 左腕部相机
      * camera_right_wrist: 右腕部相机
    - no_camera: 是否禁用相机 (default: False)
    - structure: 机器人结构类型 (default: "dualarm")
    - robot_control_hz: 控制频率 (default: 120)
    - enable_logging: 是否启用日志输出 (default: False，只输出error级别以上的日志)
    - enable_soft_realtime: 是否启用软实时调度 (default: False，需要先执行scripts/setup_rtprio.sh)
    - bind_cpu_idxs: CPU绑定索引列表 (default: None，如[5]表示绑定到第5个CPU核心)
    - disabled_cameras: 禁用的相机列表 (default: None，如["cam_high", "cam_left_wrist"])

    内部参数（用户不需要关心）：
    - log_queue: 日志队列
    - simu_mode: 仿真模式，会自动禁用真实相机 (default: False)
    - virtual_robot: 虚拟机器人模式 (default: False)

    注意：simu_mode=True或no_camera=True时会自动禁用相机功能，使用虚拟图像代替
    """
    try:
        import rclpy
        rclpy.init()
    except:
        pass

    return RealEnv(**kwargs)
