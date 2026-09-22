import enum
import logging
import json
import threading
import time
import os
import multiprocessing as mp
import queue
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import Float64MultiArray
import scipy.spatial.transform as st
import numpy as np
from mc_core_interface.msg import MechUnitCmdArray, MechUnitCmd, MechUnitState, MechUnitStateArray
from mc_core_interface.srv import RobotCmdService

from mozrobot.utils import TimingProfiler, log_drift_dt_info, realtime_precise_sleep, set_soft_realtime, set_cpu_affinity
from mozrobot.envs.moz1_controller.moz1_jnttocart import jnttocart

class Moz1MUIndexPosInList(enum.Enum):
    TORSO = 0
    LEFT_ARM = 1
    RIGHT_ARM = 2
    BASE = 3

class Moz1MUControlMode(enum.Enum):
    NOP_MODE = enum.auto()
    JOINT_CONTROL = enum.auto()
    CARTESIAN_CONTROL = enum.auto()

class TransUtils():
    @staticmethod
    def pose_msg_to_list(pose):
        return TransUtils.pose_msg_to_numpy(pose).tolist()

    @staticmethod
    def pose_msg_to_numpy(pose):
        pos = np.array([pose.end_pose.position.x, pose.end_pose.position.y, pose.end_pose.position.z])
        quat_xyzw = np.array([pose.end_pose.orientation.x, pose.end_pose.orientation.y, pose.end_pose.orientation.z, pose.end_pose.orientation.w])
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec])

    @staticmethod
    def numpy_to_pose_msg(pose: np.ndarray):
        """Convert pose to ROS Pose message.

        Args:
            pose: np.ndarray of [x, y, z, rx, ry, rz] (dtype=float32)

        Returns:
            Pose: ROS Pose message

        Note:
            Expects numpy array input. Upper layer (RealEnv) ensures type safety.
        """
        pose_msg = Pose()
        position = pose[:3]
        orientation = st.Rotation.from_rotvec(pose[3:]).as_quat()
        pose_msg.position.x = float(position[0])
        pose_msg.position.y = float(position[1])
        pose_msg.position.z = float(position[2])
        pose_msg.orientation.x = float(orientation[0])
        pose_msg.orientation.y = float(orientation[1])
        pose_msg.orientation.z = float(orientation[2])
        pose_msg.orientation.w = float(orientation[3])

        return pose_msg


class Moz1BrigeInterface(Node):
    def __init__(self, rate_hz = 120, mu_idxs_list=(None, 0, 1, None), timeout=10.0, enable_logging=False, enable_soft_realtime=False, bind_cpu_idxs=None):
        import rclpy.context
        self.ros_context = rclpy.context.Context()
        rclpy.init(context=self.ros_context)
        super().__init__('Moz1BrigeInterface', context=self.ros_context)

        # Route all logging calls in this module through a dedicated logger
        self.logger = logging.getLogger('moz1_arm_driver')

        # Configure logging level
        if not enable_logging:
            self.logger.setLevel(logging.ERROR)

        self.logger.info(f"[Moz1BrigeInterface] Started, PID: {os.getpid()}")

        self.rate_hz = rate_hz
        self.enable_soft_realtime = enable_soft_realtime
        self.bind_cpu_idxs = bind_cpu_idxs
        self.mu_list_states_lock = threading.Lock()
        self.last_cart_states_time = None  # Add timestamp tracking
        self.cart_states_timeout = 1.0  # 1 second timeout

        ros_subscriptions = []

        self.mix_command_publisher = self.create_publisher(MechUnitCmdArray, 'mx_mix_command', 10)
        self.gripper_command_publisher = self.create_publisher(Float64MultiArray, 'mx_gripper_command', 10)
        self.base_vel_command_publisher = self.create_publisher(Vector3, 'mx_base_vel_command', 10)

        # Create service client for robot commands
        self.robot_cmd_service = self.create_client(RobotCmdService, 'robot_cmd_service')

        self.cur_ee_pose_subscription = self.create_subscription(MechUnitStateArray, 'cart_states', self.cur_mech_unit_state_callback, QoSProfile(depth=5))
        self.gripper_states_subscription = self.create_subscription(Float64MultiArray, 'gripper_states', self.gripper_states_callback, QoSProfile(depth=5))
        ros_subscriptions.append(self.cur_ee_pose_subscription)
        ros_subscriptions.append(self.gripper_states_subscription)

        self.subscriptions_count = len(ros_subscriptions)

        # Idx 0: torso, Idx 1: left arm, Idx 2: Right arm
        self.mu_idxs_list = mu_idxs_list
        assert len(self.mu_idxs_list) == len(Moz1MUIndexPosInList), "Given Moz1 mu idx count wrong"

        self.left_arm_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.LEFT_ARM.value]
        assert self.left_arm_mu_idx is not None, "Left arm mu idx is not set"
        self.right_arm_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.RIGHT_ARM.value]
        assert self.right_arm_mu_idx is not None, "Right arm mu idx is not set"

        # body and wheel may not exist
        self.torso_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.TORSO.value]
        self.base_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.BASE.value]

        max_mu_num = max([mu_idx for mu_idx in self.mu_idxs_list if mu_idx is not None])
        mu_list_len = max_mu_num + 1

        # State lists for get_states() method
        self.joint_pos_list = [None] * mu_list_len
        self.joint_trq_list = [None] * mu_list_len
        self.joint_vel_list = [None] * mu_list_len
        self.robot_arm_psi_list = [None] * mu_list_len
        self.robot_ee_pose_list = [None] * mu_list_len

        # Gripper state management (ROS topic-based)
        self.gripper_states_lock = threading.Lock()
        self._cached_gripper_states = [0.0, 0.0]  # [left, right] gripper positions
        self._last_gripper_states_time = None
        self._last_left_gripper_pos = np.array([0.0], dtype=np.float32)
        self._last_right_gripper_pos = np.array([0.0], dtype=np.float32)
        self.gripper_states_timeout = 1.0  # 1 second timeout

        # Start ROS spinning thread
        self.spin_thread = threading.Thread(target=self.ros_spin_thread, args=(rate_hz,))
        self.spin_thread.daemon = True
        self.spin_thread.start()

        self.logger.info("wait for robot to be alive")
        # Wait for robot to be alive
        if not self.wait_for_robot_alive(timeout):
            raise RuntimeError("Robot is not alive after waiting for {} seconds".format(timeout))
        self.logger.info("robot is alive")

    def get_states(self):
        """Get all available robot states in a unified dictionary format.

        Returns:
            dict: Dictionary containing available states (no None values):
                - left_arm_joints: list of joint positions
                - right_arm_joints: list of joint positions  
                - left_arm_ee_pose: list of cartesian pose [x,y,z,rx,ry,rz]
                - right_arm_ee_pose: list of cartesian pose [x,y,z,rx,ry,rz]
                - left_arm_psi: float PSI value
                - right_arm_psi: float PSI value
                - joint_velocities: dict with left_arm/right_arm/torso keys
                - joint_torques: dict with left_arm/right_arm/torso keys
                - base_speed: list of base speed [x,y,z]
                - torso_joints: list of joint positions (only if torso exists)
                - torso_ee_pose: list of cartesian pose (only if torso exists)
        """
        states = {}

        with self.mu_list_states_lock:
            # Always present: dual arms
            joint_positions = self.joint_pos_list
            joint_velocities = self.joint_vel_list
            joint_torques = self.joint_trq_list
            ee_poses = self.robot_ee_pose_list
            psi_values = self.robot_arm_psi_list

            if (joint_positions[self.left_arm_mu_idx] is not None and 
                joint_positions[self.right_arm_mu_idx] is not None):
                states['left_arm_joints'] = joint_positions[self.left_arm_mu_idx]
                states['right_arm_joints'] = joint_positions[self.right_arm_mu_idx]

            if (ee_poses[self.left_arm_mu_idx] is not None and 
                ee_poses[self.right_arm_mu_idx] is not None):
                states['left_arm_ee_pose'] = ee_poses[self.left_arm_mu_idx]
                states['right_arm_ee_pose'] = ee_poses[self.right_arm_mu_idx]

            if (psi_values[self.left_arm_mu_idx] is not None and 
                psi_values[self.right_arm_mu_idx] is not None):
                states['left_arm_psi'] = psi_values[self.left_arm_mu_idx]
                states['right_arm_psi'] = psi_values[self.right_arm_mu_idx]

            # Velocities and torques
            velocity_dict = {}
            torque_dict = {}

            if joint_velocities[self.left_arm_mu_idx] is not None:
                velocity_dict['left_arm'] = joint_velocities[self.left_arm_mu_idx]
            if joint_velocities[self.right_arm_mu_idx] is not None:
                velocity_dict['right_arm'] = joint_velocities[self.right_arm_mu_idx]
            if joint_torques[self.left_arm_mu_idx] is not None:
                torque_dict['left_arm'] = joint_torques[self.left_arm_mu_idx]
            if joint_torques[self.right_arm_mu_idx] is not None:
                torque_dict['right_arm'] = joint_torques[self.right_arm_mu_idx]

            # Optional: torso
            if (self.torso_mu_idx is not None and 
                joint_positions[self.torso_mu_idx] is not None):
                states['torso_joints'] = joint_positions[self.torso_mu_idx]
                if joint_velocities[self.torso_mu_idx] is not None:
                    velocity_dict['torso'] = joint_velocities[self.torso_mu_idx]
                if joint_torques[self.torso_mu_idx] is not None:
                    torque_dict['torso'] = joint_torques[self.torso_mu_idx]

            if (self.torso_mu_idx is not None and 
                ee_poses[self.torso_mu_idx] is not None):
                states['torso_ee_pose'] = ee_poses[self.torso_mu_idx]

            if (self.base_mu_idx is not None and 
                ee_poses[self.base_mu_idx] is not None):
                states['base_speed'] = ee_poses[self.base_mu_idx][:3]

            if velocity_dict:
                states['joint_velocities'] = velocity_dict
            if torque_dict:
                states['joint_torques'] = torque_dict

        # Add gripper states from ROS topic cache
        with self.gripper_states_lock:
            if len(self._cached_gripper_states) >= 2:
                states['left_gripper_position'] = self._cached_gripper_states[0]
                states['right_gripper_position'] = self._cached_gripper_states[1]

        return states

    def send_action(self, action: dict) -> None:
        """Send unified action command to the robot.

        Args:
            action: Dictionary with optional keys:
                - left_arm_joints: np.ndarray of joint positions (shape: (7,), dtype=float32)
                - right_arm_joints: np.ndarray of joint positions (shape: (7,), dtype=float32)
                - left_arm_ee_pose: np.ndarray of cartesian pose [x,y,z,rx,ry,rz] (shape: (6,), dtype=float32)
                - right_arm_ee_pose: np.ndarray of cartesian pose [x,y,z,rx,ry,rz] (shape: (6,), dtype=float32)
                - torso_joints: np.ndarray of joint positions (shape: (6,), dtype=float32)
                - torso_ee_pose: np.ndarray of cartesian pose (shape: (6,), dtype=float32)
                - base_cmd: np.ndarray [x,y,z] velocity (shape: (3,), dtype=float32)
                - left_gripper_position: np.ndarray gripper position (shape: (1,), dtype=float32)
                - right_gripper_position: np.ndarray gripper position (shape: (1,), dtype=float32)

        Priority: joint commands override cartesian commands if both present.

        Note: This is an internal driver interface. All inputs must be np.ndarray (dtype=float32).
              Upper layer (RealEnv) is responsible for converting user inputs to np.ndarray.
        """
        if not action:
            return

        # Handle base control via ROS topic
        if 'base_cmd' in action:
            base_cmd = action['base_cmd']
            base_vel_msg = Vector3()
            base_vel_msg.x = float(base_cmd[0])  # x方向速度
            base_vel_msg.y = float(base_cmd[1])  # y方向速度 (根据原有逻辑)
            base_vel_msg.z = float(base_cmd[2])  # 绕z轴旋转速度

            self.base_vel_command_publisher.publish(base_vel_msg)
            self.logger.debug(f"Published base velocity command: x={base_cmd[0]:.3f}, y={base_cmd[1]:.3f}, z={base_cmd[2]:.3f}")

        # Handle arm and torso control - joint commands take priority
        movax_control_cmds = MechUnitCmdArray()
        movax_control_cmds.header.stamp = self.get_clock().now().to_msg()

        control_configs = [
            ('left_arm', self.left_arm_mu_idx),
            ('right_arm', self.right_arm_mu_idx),
            ('torso', self.torso_mu_idx),
        ]

        # joint action: left_arm_joints, right_arm_joints, torso_joints
        # cartesian action: left_arm_ee_pose, right_arm_ee_pose, torso_ee_pose
        for prefix, mu_idx in control_configs:
            joint_key = f'{prefix}_joints'
            ee_pose_key = f'{prefix}_ee_pose'

            if joint_key in action:
                self._send_joint_command(movax_control_cmds, mu_idx, action[joint_key])
            elif ee_pose_key in action:
                # No need to convert - already np.ndarray from _prepare_action
                self._send_cartesian_command(movax_control_cmds, mu_idx, action[ee_pose_key])

        # Handle gripper commands via ROS topic
        if 'left_gripper_position' in action or 'right_gripper_position' in action:
            gripper_msg = Float64MultiArray()
            left_pos = action.get('left_gripper_position', self._last_left_gripper_pos)
            right_pos = action.get('right_gripper_position', self._last_right_gripper_pos)
            # Extract scalar value from np.ndarray if needed
            left_val = float(left_pos[0]) if isinstance(left_pos, np.ndarray) else float(left_pos)
            right_val = float(right_pos[0]) if isinstance(right_pos, np.ndarray) else float(right_pos)
            gripper_msg.data = [left_val, right_val]
            self.gripper_command_publisher.publish(gripper_msg)

            # Update last positions for next time (store as np.ndarray for consistency)
            self._last_left_gripper_pos = left_pos
            self._last_right_gripper_pos = right_pos

            self.logger.debug(f"Published gripper command: left={left_val:.4f}, right={right_val:.4f}")

        # Publish commands directly
        if len(movax_control_cmds.cmds) > 0:
            self.mix_command_publisher.publish(movax_control_cmds)

    def _send_joint_command(self, movax_control_cmds: MechUnitCmdArray, mu_idx: int, joint_positions: np.ndarray):
        """Send joint control command directly.

        Args:
            movax_control_cmds: Command array to append to
            mu_idx: Mechanical unit index
            joint_positions: np.ndarray of joint positions (dtype=float32)

        Note:
            Expects numpy array input. Upper layer (RealEnv) ensures type safety.
        """
        if mu_idx is None:
            return

        mu_cmd = MechUnitCmd()
        mu_cmd.mu_idx = mu_idx
        mu_cmd.use_jnt = True
        # Convert numpy array to list for ROS2 message
        mu_cmd.jnt_pos = joint_positions.tolist()
        movax_control_cmds.cmds.append(mu_cmd)

    def _send_cartesian_command(self, movax_control_cmds: MechUnitCmdArray, mu_idx: int, target_pose: np.ndarray):
        """Send cartesian control command directly.

        Args:
            movax_control_cmds: Command array to append to
            mu_idx: Mechanical unit index
            target_pose: np.ndarray of cartesian pose [x,y,z,rx,ry,rz] (dtype=float32)
        """
        if mu_idx is None:
            return

        mu_cmd = MechUnitCmd()
        mu_cmd.mu_idx = mu_idx
        mu_cmd.use_jnt = False
        mu_cmd.end_pose = TransUtils.numpy_to_pose_msg(target_pose)

        # Get current PSI value for cartesian control
        psi_value = 0.0  # Default PSI
        with self.mu_list_states_lock:
            if (self.robot_arm_psi_list and
                mu_idx < len(self.robot_arm_psi_list) and
                self.robot_arm_psi_list[mu_idx] is not None):
                psi_value = self.robot_arm_psi_list[mu_idx]

        mu_cmd.psi = psi_value
        movax_control_cmds.cmds.append(mu_cmd)

    def reset_robot_positions(self, control_info: dict):
        self.disable_external_following_mode()

        if "left_arm_joints" in control_info:
            if len(control_info["left_arm_joints"]) == 7 and self.left_arm_mu_idx is not None:
                self._jog_mu(self.left_arm_mu_idx, control_info["left_arm_joints"])
            else:
                self.logger.error(f"Left arm position length is not 7: {len(control_info['left_arm'])}")
        if "right_arm_joints" in control_info:
            if len(control_info["right_arm_joints"]) == 7 and self.right_arm_mu_idx is not None:
                self._jog_mu(self.right_arm_mu_idx, control_info["right_arm_joints"])
            else:
                self.logger.error(f"Right arm position length is not 7: {len(control_info['right_arm'])}")
        if "torso_joints" in control_info:
            if len(control_info["torso_joints"]) == 6 and self.torso_mu_idx is not None:
                self._jog_mu(self.torso_mu_idx, control_info["torso_joints"])
            else:
                self.logger.error(f"Torso position length is not 6: {len(control_info['torso'])}")
        if "gripper_positions" in control_info:
            if len(control_info["gripper_positions"]) == 2:
                self.reset_gripper_pos(control_info["gripper_positions"])
            else:
                self.logger.error(f"Gripper position length is not 2: {len(control_info['gripper'])}")

    def _jog_mu(self, mu_idx: int, joint_pos: List[float]):
        jog_mu_ = RobotCmdService.Request()
        jog_mu_.cmd = "JOGMechUnit"

        # rad -> deg, since movax JOGMechUnit use deg
        joint_pos = [pos * 180 / np.pi for pos in joint_pos]
        jog_mu_.data = json.dumps({"mech_idx": mu_idx, "mu_target_pos": joint_pos})
        self.future = self.robot_cmd_service.call_async(jog_mu_)
        if not self.future:
            self.logger.error('Failed to jog mech unit')
        else:
            self.logger.info('Service call jog mech unit successful')

    def move_robot_to_workpoint(self, workpoint_name: str):
        self.disable_external_following_mode()

        """Move robot to workpoint."""
        move_to_workpoint_ = RobotCmdService.Request()
        move_to_workpoint_.cmd = "WorkPointMoveTo"

        enable_mechunits = []
        if self.left_arm_mu_idx is not None:
            enable_mechunits.append("LeftArm")
        if self.right_arm_mu_idx is not None:
            enable_mechunits.append("RightArm")
        if self.torso_mu_idx is not None:
            enable_mechunits.append("LegWaist")

        move_to_workpoint_.data = json.dumps({"workpoint_name": workpoint_name, "mech_unit": enable_mechunits})
        self.future = self.robot_cmd_service.call_async(move_to_workpoint_)
        if not self.future:
            logging.error('Failed to move to workpoint')
        else:
            logging.info('Service call move to workpoint successful')

    def enable_external_following_mode(self):
        enable_outer_ctrl_ = RobotCmdService.Request()
        enable_outer_ctrl_.cmd = "EnableOuterCtrl"
        enable_outer_ctrl_.data = json.dumps({"enable": True})
        self.future = self.robot_cmd_service.call_async(enable_outer_ctrl_)
        if not self.future:
            logging.error('Failed to enable outer control')
        else:
            logging.info('Service call enable outer control successful')

        switch_to_mix_tracking_ = RobotCmdService.Request()
        switch_to_mix_tracking_.cmd = "SwitchToMixTrackingMode"
        self.future = self.robot_cmd_service.call_async(switch_to_mix_tracking_)
        if not self.future:
            logging.error('Failed to switch to mix tracking mode')
        else:
            logging.info('Service call switch to mix tracking mode successful')

    def disable_external_following_mode(self):
        disable_outer_ctrl_ = RobotCmdService.Request()
        disable_outer_ctrl_.cmd = "EnableOuterCtrl"
        disable_outer_ctrl_.data = json.dumps({"enable": False})
        self.future = self.robot_cmd_service.call_async(disable_outer_ctrl_)
        if not self.future:
            logging.error('Failed to disable outer control')
        else:
            logging.info('Service call disable outer control successful')

    def cur_mech_unit_state_callback(self, msg):
        callback_start = time.perf_counter()
        self.last_cart_states_time = callback_start  # Update timestamp

        try:
            for idx in range(len(msg.states)):
                cur_mu_idx = msg.states[idx].mu_idx
                if cur_mu_idx not in self.mu_idxs_list:
                    continue

                cur_mu_state = msg.states[idx]

                robot_ee_pose = TransUtils.pose_msg_to_list(cur_mu_state)
                joint_pos = list(cur_mu_state.jnt_pos)
                # todo: mechunit state have no jnt vel feedback
                joint_vel = [ 0.0 ] * len(joint_pos)
                joint_trq = list(cur_mu_state.jnt_trq)
                arm_psi = cur_mu_state.psi

                # Use timeout for lock acquisition to prevent callback from blocking
                lock_acquired = False
                try:
                    lock_acquired = self.mu_list_states_lock.acquire(timeout=0.05)  # 50ms timeout
                    if lock_acquired:
                        self.joint_pos_list[cur_mu_idx] = joint_pos
                        self.joint_vel_list[cur_mu_idx] = joint_vel
                        self.joint_trq_list[cur_mu_idx] = joint_trq
                        self.robot_arm_psi_list[cur_mu_idx] = arm_psi
                        self.robot_ee_pose_list[cur_mu_idx] = robot_ee_pose
                    else:
                        self.logger.warning(f"Callback failed to acquire mu_list_states_lock within timeout")
                except Exception as e:
                    self.logger.error(f"Error in callback state update: {e}")
                finally:
                    if lock_acquired:
                        try:
                            self.mu_list_states_lock.release()
                        except:
                            pass

        except Exception as e:
            self.logger.error(f"Error in cur_mech_unit_state_callback: {e}")

        callback_time = time.perf_counter() - callback_start
        if callback_time > 1 / self.rate_hz:
            self.logger.warning(f"State callback took {callback_time:.4f}s, which is unusually long")

    def gripper_states_callback(self, msg):
        """Callback for gripper states from ROS topic."""
        callback_start = time.perf_counter()

        try:
            if len(msg.data) >= 2:
                self._last_gripper_states_time = callback_start
                with self.gripper_states_lock:
                    self._cached_gripper_states = msg.data[:2]  # [left, right] gripper positions
                    self.logger.debug(f"Received gripper states: left={self._cached_gripper_states[0]:.4f}, right={self._cached_gripper_states[1]:.4f}")
            else:
                self.logger.warning(f"Gripper states message has insufficient data: {len(msg.data)} elements, expected at least 2")
        except Exception as e:
            self.logger.error(f"Error in gripper_states_callback: {e}")

        callback_time = time.perf_counter() - callback_start
        if callback_time > 1 / self.rate_hz:
            self.logger.warning(f"Gripper callback took {callback_time:.4f}s, which is unusually long")

    def is_gripper_alive(self):
        """Check if gripper states are being received within timeout."""
        if self._last_gripper_states_time is None:
            return False
        current_time = time.perf_counter()
        return (current_time - self._last_gripper_states_time) < self.gripper_states_timeout

    def ros_spin_thread(self, rate_hz):
        """ROS spinning thread for processing messages only."""
        if self.enable_soft_realtime:
            set_soft_realtime(priority=20, thread_id="moz1_ros_spin")
        if self.bind_cpu_idxs is not None:
            set_cpu_affinity(cpu_idxs=list(self.bind_cpu_idxs), thread_id="moz1_ros_spin")

        period_sec = 1 / rate_hz
        tp = TimingProfiler()

        while rclpy.ok(context=self.ros_context):
            start_loop_time = time.perf_counter()
            end_time = start_loop_time + period_sec

            tp.start("ros_spin_once")
            try:
                # Process multiple messages per cycle to handle increased callback load
                # while maintaining 120Hz frequency
                for _ in range(self.subscriptions_count):
                    rclpy.spin_once(self, timeout_sec=0.001)  # Reduced timeout to 1ms
            except Exception as e:
                self.logger.error(f"spin_once failed: {e}")
                continue
            tp.stop("ros_spin_once")

            remain_time = end_time - time.perf_counter()
            if remain_time > 0:
                tp.start("sleep_time")
                realtime_precise_sleep(remain_time)
                tp.stop("sleep_time")
                tp.add_to_timing("time_to_sleep", remain_time)

            dt_s = time.perf_counter() - start_loop_time
            log_drift_dt_info(dt_s, rate_hz, id="moz1_ros_spin", profiler=tp, drift_threshold=0.1)

            # Log performance statistics every 1 second
            if tp.get_timing_count("ros_spin_once") >= rate_hz * 1:
                avg_ros_spin = tp.get_timing_average("ros_spin_once")
                max_ros_spin = tp.get_timing_max("ros_spin_once")

                if max_ros_spin > period_sec:
                    self.logger.warning(
                        f"ROS spin performance: avg={avg_ros_spin:.4f}s max={max_ros_spin:.4f}s"
                    )

                tp.reset()

        self.logger.warning("rclpy not ready, Moz1ArmDriver sub_thread_entry will exit...s")

    def is_robot_alive(self):
        """Check if the robot is alive by verifying cart_states feedback"""
        if self.last_cart_states_time is None:
            return False
        current_time = time.perf_counter()
        return (current_time - self.last_cart_states_time) < self.cart_states_timeout

    def wait_for_robot_alive(self, timeout=10.0):
        """Wait for robot to be alive for a specified timeout period
        Args:
            timeout (float): Maximum time to wait in seconds
        Returns:
            bool: True if robot became alive within timeout, False otherwise
        """
        start_time = time.perf_counter()
        rate = 0.1  # Check every 100ms

        while (time.perf_counter() - start_time) < timeout:
            if self.is_robot_alive():
                wait_time = time.perf_counter() - start_time
                self.logger.info(f"Robot is alive and responding after {wait_time} seconds")
                return True
            time.sleep(rate)

        self.logger.error("Robot failed to respond within {} seconds".format(timeout))
        return False

    def reset_gripper_pos(self, gripper_pos: List[float]):
        """Reset grippers to home position via ROS topic.

        Args:
            gripper_pos: List of [left_pos, right_pos] gripper positions
        """
        gripper_msg = Float64MultiArray()
        gripper_msg.data = gripper_pos
        self.gripper_command_publisher.publish(gripper_msg)

        # Update last positions (store as np.ndarray for consistency)
        self._last_left_gripper_pos = np.array([gripper_pos[0]], dtype=np.float32)
        self._last_right_gripper_pos = np.array([gripper_pos[1]], dtype=np.float32)

        self.logger.info("Reset grippers to home position via ROS topic")

class Moz1BrigeInterfaceVirtual:
    def __init__(self, rate_hz = 120, mu_idxs_list=(None, 0, 1, None), timeout=10.0, enable_logging=False, enable_soft_realtime=False, bind_cpu_idxs=None):
        self.rate_hz = rate_hz
        self.enable_soft_realtime = enable_soft_realtime
        self.bind_cpu_idxs = bind_cpu_idxs
        self.mu_list_states_lock = threading.Lock()

        # Route all logging calls in this module through a dedicated logger
        self.logger = logging.getLogger('moz1_arm_driver')

        # Configure logging level
        if not enable_logging:
            self.logger.setLevel(logging.ERROR)

        # Idx 0: torso, Idx 1: left arm, Idx 2: Right arm
        self.mu_idxs_list = mu_idxs_list
        assert len(self.mu_idxs_list) == len(Moz1MUIndexPosInList), "Given Moz1 mu idx count wrong"

        self.left_arm_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.LEFT_ARM.value]
        assert self.left_arm_mu_idx is not None, "Left arm mu idx is not set"
        self.right_arm_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.RIGHT_ARM.value]
        assert self.right_arm_mu_idx is not None, "Right arm mu idx is not set"

        # body and wheel may not exist
        self.torso_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.TORSO.value]
        self.base_mu_idx = self.mu_idxs_list[Moz1MUIndexPosInList.BASE.value]

        max_mu_num = max([mu_idx for mu_idx in self.mu_idxs_list if mu_idx is not None])
        mu_list_len = max_mu_num + 1

        self.joint_pos_list = [None] * mu_list_len
        self.joint_trq_list = [None] * mu_list_len
        self.joint_vel_list = [None] * mu_list_len
        self.robot_arm_psi_list = [0.0] * mu_list_len
        self.robot_ee_pose_list = [None] * mu_list_len

        self.mu_control_lock = [threading.Lock() for _ in range(mu_list_len)]
        self.mu_control_mode = [Moz1MUControlMode.NOP_MODE] * mu_list_len
        
        # Initialize gripper controllers (virtual - just placeholders)
        self.gripper_scale = 1
        self.left_gripper = None
        self.right_gripper = None
        
        self.logger = logging.getLogger('moz1_arm_driver_virtual')
        self.logger.info("virtual robot is alive")

    def get_states(self):
        """Get all available robot states in a unified dictionary format.
        
        Returns:
            dict: Dictionary containing available states (no None values):
                - left_arm_joints: list of joint positions
                - right_arm_joints: list of joint positions  
                - left_arm_ee_pose: list of cartesian pose [x,y,z,rx,ry,rz]
                - right_arm_ee_pose: list of cartesian pose [x,y,z,rx,ry,rz]
                - left_arm_psi: float PSI value
                - right_arm_psi: float PSI value
                - joint_velocities: dict with left_arm/right_arm/torso keys
                - joint_torques: dict with left_arm/right_arm/torso keys
                - torso_joints: list of joint positions (only if torso exists)
                - torso_ee_pose: list of cartesian pose (only if torso exists)
        """
        states = {}
        
        with self.mu_list_states_lock:
            # Always present: dual arms
            joint_positions = self.joint_pos_list
            joint_velocities = self.joint_vel_list
            joint_torques = self.joint_trq_list
            ee_poses = self.robot_ee_pose_list
            psi_values = self.robot_arm_psi_list
            
            if (joint_positions[self.left_arm_mu_idx] is not None and 
                joint_positions[self.right_arm_mu_idx] is not None):
                states['left_arm_joints'] = joint_positions[self.left_arm_mu_idx]
                states['right_arm_joints'] = joint_positions[self.right_arm_mu_idx]
                
            if (ee_poses[self.left_arm_mu_idx] is not None and 
                ee_poses[self.right_arm_mu_idx] is not None):
                states['left_arm_ee_pose'] = ee_poses[self.left_arm_mu_idx]
                states['right_arm_ee_pose'] = ee_poses[self.right_arm_mu_idx]
                
            if (psi_values[self.left_arm_mu_idx] is not None and 
                psi_values[self.right_arm_mu_idx] is not None):
                states['left_arm_psi'] = psi_values[self.left_arm_mu_idx]
                states['right_arm_psi'] = psi_values[self.right_arm_mu_idx]
            
            # Velocities and torques
            velocity_dict = {}
            torque_dict = {}
            
            if joint_velocities[self.left_arm_mu_idx] is not None:
                velocity_dict['left_arm'] = joint_velocities[self.left_arm_mu_idx]
            if joint_velocities[self.right_arm_mu_idx] is not None:
                velocity_dict['right_arm'] = joint_velocities[self.right_arm_mu_idx]
            if joint_torques[self.left_arm_mu_idx] is not None:
                torque_dict['left_arm'] = joint_torques[self.left_arm_mu_idx]
            if joint_torques[self.right_arm_mu_idx] is not None:
                torque_dict['right_arm'] = joint_torques[self.right_arm_mu_idx]
                
            # Optional: torso
            if (self.torso_mu_idx is not None and 
                joint_positions[self.torso_mu_idx] is not None):
                states['torso_joints'] = joint_positions[self.torso_mu_idx]
                if joint_velocities[self.torso_mu_idx] is not None:
                    velocity_dict['torso'] = joint_velocities[self.torso_mu_idx]
                if joint_torques[self.torso_mu_idx] is not None:
                    torque_dict['torso'] = joint_torques[self.torso_mu_idx]
                
            if (self.torso_mu_idx is not None and 
                ee_poses[self.torso_mu_idx] is not None):
                states['torso_ee_pose'] = ee_poses[self.torso_mu_idx]
                
            if velocity_dict:
                states['joint_velocities'] = velocity_dict
            if torque_dict:
                states['joint_torques'] = torque_dict
                
            # Add simulated gripper states (fixed values for virtual interface)
            states['left_gripper_position'] = 0.0
            states['right_gripper_position'] = 0.0
                
        return states

    def send_action(self, action: dict) -> None:
        """Send unified action command to the virtual robot.

        Args:
            action: Dictionary with optional keys:
                - left_arm_joints: np.ndarray of joint positions (shape: (7,), dtype=float32)
                - right_arm_joints: np.ndarray of joint positions (shape: (7,), dtype=float32)
                - left_arm_ee_pose: np.ndarray of cartesian pose [x,y,z,rx,ry,rz] (shape: (6,), dtype=float32)
                - right_arm_ee_pose: np.ndarray of cartesian pose [x,y,z,rx,ry,rz] (shape: (6,), dtype=float32)
                - torso_joints: np.ndarray of joint positions (shape: (6,), dtype=float32)
                - torso_ee_pose: np.ndarray of cartesian pose (shape: (6,), dtype=float32)
                - base_cmd: np.ndarray [x,y,z] velocity (shape: (3,), dtype=float32) (logged but no action taken)
                - left_gripper_position: np.ndarray gripper position (shape: (1,), dtype=float32) (logged but no action taken)
                - right_gripper_position: np.ndarray gripper position (shape: (1,), dtype=float32) (logged but no action taken)

        Priority: joint commands override cartesian commands if both present.

        Note: This is an internal driver interface. All inputs must be np.ndarray (dtype=float32).
              Upper layer (RealEnv) is responsible for converting user inputs to np.ndarray.
        """
        if not action:
            return

        # Handle base control (virtual - just logging)
        if 'base_cmd' in action:
            base_cmd = action['base_cmd']
            move_dir = base_cmd[0]
            trans_vel = base_cmd[1]
            rot_vel = base_cmd[2]
            if (move_dir == 0 and trans_vel == 0 and rot_vel == 0):
                self.logger.info("Virtual robot: stop moving base")
            else:
                self.logger.info(f"Virtual robot: moving base in direction: {move_dir}, speed: {trans_vel}, rot_speed: {rot_vel}")

        # Handle arm control - joint commands take priority
        if 'left_arm_joints' in action and 'right_arm_joints' in action:
            # Use joint control
            self.__update_desired_mu_joint_pos(self.left_arm_mu_idx, action['left_arm_joints'])
            self.__update_desired_mu_joint_pos(self.right_arm_mu_idx, action['right_arm_joints'])
        else:
            # Use cartesian control - no need to convert, already np.ndarray
            if 'left_arm_ee_pose' in action:
                self.__update_desired_mu_ee_pose(self.left_arm_mu_idx, action['left_arm_ee_pose'])
            if 'right_arm_ee_pose' in action:
                self.__update_desired_mu_ee_pose(self.right_arm_mu_idx, action['right_arm_ee_pose'])
        
        # Handle torso - joint commands take priority
        if 'torso_joints' in action:
            self.__update_desired_mu_joint_pos(self.torso_mu_idx, action['torso_joints'])
        elif 'torso_ee_pose' in action:
            # No need to convert - already np.ndarray
            self.__update_desired_mu_ee_pose(self.torso_mu_idx, action['torso_ee_pose'])
        
        # Handle gripper commands (virtual - just logging)
        if 'left_gripper_position' in action:
            position = action['left_gripper_position']
            self.logger.info(f"Virtual robot: set left gripper position to {position}")
            
        if 'right_gripper_position' in action:
            position = action['right_gripper_position']
            self.logger.info(f"Virtual robot: set right gripper position to {position}")

    def __arm_jnt_to_cart(self, jnt_angles: np.ndarray, is_left: bool) -> np.ndarray:
        """
        Calculate cartesian pose from joint angles.
        
        Args:
            jnt_angles: joint angles
            is_left: True for left arm, False for right arm
            
        Returns:
            Cartesian pose as numpy array
        """

        if len(jnt_angles) != 7:
            self.logger.error(f"jnt_angles shape is {len(jnt_angles)}, expected 7")
            return None

        # Calculate end-effector poses using forward kinematics
        cart_pose_raw = jnttocart(jnt_angles[0], jnt_angles[1], jnt_angles[2], jnt_angles[3], jnt_angles[4], jnt_angles[5], jnt_angles[6], is_left)
        
        # Convert roll, pitch, yaw to rotation vectors
        cart_rot_vec = st.Rotation.from_euler('xyz', cart_pose_raw[3:]).as_rotvec()

        # Create cartesian pose
        cart_pose = np.asarray(np.concatenate((cart_pose_raw[:3], cart_rot_vec)), dtype=np.float32)
        
        return cart_pose
    
    def __torso_jnt_to_cart(self, jnt_angles: np.ndarray) -> np.ndarray:
        """
        Calculate cartesian pose from joint angles.
        
        Args:
            jnt_angles: joint angles
            is_left: True for left arm, False for right arm
            
        Returns:
            Cartesian pose as numpy array
        """

        if len(jnt_angles) != 6:
            self.logger.error(f"jnt_angles shape is {len(jnt_angles)}, expected 6")
            return None
                
        # Create cartesian pose, todo: implement
        cart_pose = [0.0] * 6
        
        return cart_pose

    def __update_desired_mu_ee_pose(self, mu_idx: int, target_pose: np.ndarray):
        assert mu_idx is not None, "Mu idx is not set"
        with self.mu_control_lock[mu_idx]:
            mu_control_mode = self.mu_control_mode[mu_idx]

            if mu_control_mode != Moz1MUControlMode.CARTESIAN_CONTROL:
                self.mu_control_mode[mu_idx] = Moz1MUControlMode.CARTESIAN_CONTROL

        with self.mu_list_states_lock:
            if mu_idx == self.left_arm_mu_idx or mu_idx == self.right_arm_mu_idx:
                self.joint_pos_list[mu_idx] = [0.0] * 7
                self.joint_vel_list[mu_idx] = [0.0] * 7
                self.joint_trq_list[mu_idx] = [0.0] * 7
                self.robot_arm_psi_list[mu_idx] = 0.0
                self.robot_ee_pose_list[mu_idx] = target_pose
            elif mu_idx == self.torso_mu_idx:
                self.joint_pos_list[mu_idx] = [0.0] * 6
                self.joint_vel_list[mu_idx] = [0.0] * 6
                self.joint_trq_list[mu_idx] = [0.0] * 6
                self.robot_arm_psi_list[mu_idx] = 0.0
                self.robot_ee_pose_list[mu_idx] = target_pose
            else:
                self.logger.error(f"mu_idx {mu_idx} is not left or right arm or torso")
                return

    def __update_desired_mu_joint_pos(self, mu_idx: int, target_joint_pos):
        assert mu_idx is not None, "Mu idx is not set"
        with self.mu_control_lock[mu_idx]:
            mu_control_mode = self.mu_control_mode[mu_idx]
            self.mu_control_mode[mu_idx] = Moz1MUControlMode.JOINT_CONTROL

        with self.mu_list_states_lock:
            self.joint_pos_list[mu_idx] = target_joint_pos
            if mu_idx == self.left_arm_mu_idx or mu_idx == self.right_arm_mu_idx:
                self.joint_vel_list[mu_idx] = [0.0] * 7
                self.joint_trq_list[mu_idx] = [0.0] * 7
                self.robot_arm_psi_list[mu_idx] = 0.0
                self.robot_ee_pose_list[mu_idx] = self.__arm_jnt_to_cart(target_joint_pos, mu_idx == self.left_arm_mu_idx)
            elif mu_idx == self.torso_mu_idx:
                self.joint_vel_list[mu_idx] = [0.0] * 6
                self.joint_trq_list[mu_idx] = [0.0] * 6
                self.robot_arm_psi_list[mu_idx] = 0.0
                self.robot_ee_pose_list[mu_idx] = self.__torso_jnt_to_cart(target_joint_pos)
            else:
                self.logger.error(f"mu_idx {mu_idx} is not left or right arm or torso")
                return

    def is_robot_alive(self):
        """Check if the robot is alive by verifying cart_states feedback"""
        return True

    def wait_for_robot_alive(self, timeout=10.0):
        """Wait for robot to be alive for a specified timeout period
        Args:
            timeout (float): Maximum time to wait in seconds
        Returns:
            bool: True if robot became alive within timeout, False otherwise
        """
        return True

    def reset_robot_positions(self, control_info: dict):
        """Reset robot positions (virtual - just logging)."""
        self.logger.info(f"Virtual robot: reset robot positions to {control_info}")

    def enable_external_following_mode(self):
        """Enable external following mode (virtual - just logging)."""
        self.logger.info("Virtual robot: enable external following mode")

    def disable_external_following_mode(self):
        """Disable external following mode (virtual - just logging)."""
        self.logger.info("Virtual robot: disable external following mode")

    def set_gripper_pos(self, gripper_pos: List[float]):
        """Reset grippers to home position (virtual - just logging)."""
        self.logger.info(f"Virtual robot: set grippers to {gripper_pos}")

class _Moz1BrigeInterfaceServer:
    def __init__(self, rate_hz=120, mu_idxs_list=(None, 0, 1, None), timeout=10.0, virtual=False, simu_mode=False, log_queue=None, enable_logging=False, enable_soft_realtime=False, bind_cpu_idxs=None):
        """
        Server process that manages the actual Moz1BrigeInterface instance.

        Args:
            rate_hz: Control frequency
            mu_idxs_list: Mechanical unit indices
            timeout: Initialization timeout
            virtual: Whether to use virtual robot interface
            simu_mode: Whether to use simulation mode for grippers
            log_queue: Queue for logging
            enable_logging: Whether to enable logging below error level (default: False)
            enable_soft_realtime: Enable soft real-time scheduling (default: False)
            bind_cpu_idxs: CPU indices to bind the ROS thread to (default: None)
        """
        if log_queue is not None:
            try:
                from mozrobot.utils import setup_child_process_logging
                setup_child_process_logging(log_queue)
            except Exception:
                pass

        # Configure logging level based on enable_logging
        if not enable_logging:
            logging.getLogger('moz1_arm_driver').setLevel(logging.ERROR)
            logging.getLogger().setLevel(logging.ERROR)

        if virtual:
            self._interface = Moz1BrigeInterfaceVirtual(
                rate_hz=rate_hz,
                mu_idxs_list=mu_idxs_list,
                timeout=timeout,
                enable_logging=enable_logging,
                enable_soft_realtime=enable_soft_realtime,
                bind_cpu_idxs=bind_cpu_idxs
            )
        else:
            self._interface = Moz1BrigeInterface(
                rate_hz=rate_hz,
                mu_idxs_list=mu_idxs_list,
                timeout=timeout,
                enable_logging=enable_logging,
                enable_soft_realtime=enable_soft_realtime,
                bind_cpu_idxs=bind_cpu_idxs
            )
        self._rate_hz = rate_hz

    def _state_thread(self, state_queue: mp.Queue):
        """Thread to continuously publish robot states."""
        try:
            while True:
                state_time = time.perf_counter()
                states = self._interface.get_states()

                # Put state in queue, drop old states if queue is full
                while True:
                    try:
                        state_queue.put_nowait((states, state_time))
                        break
                    except queue.Full:
                        try:
                            # Remove old state to make room for new one
                            state_queue.get_nowait()
                        except queue.Empty:
                            pass

                # Sleep to maintain reasonable state publishing frequency
                # State updates don't need to be as fast as control loop
                realtime_precise_sleep(state_time + 1.0 / self._rate_hz - time.perf_counter())
        except KeyboardInterrupt:
            logging.info("State thread interrupted by user")
            return

    def run(self, state_queue: mp.Queue, action_queue: mp.Queue, response_queue: mp.Queue):
        """
        Main server loop for processing actions and maintaining stable control frequency.

        Args:
            state_queue: Queue for sending states to client
            action_queue: Queue for receiving actions from client
            response_queue: Queue for sending responses to sync operations
        """
        logger = logging.getLogger("moz1_srv_run")

        # Start state publishing thread
        state_thread = threading.Thread(target=self._state_thread, args=(state_queue,))
        state_thread.daemon = True
        state_thread.start()
        
        # Main control loop
        action_buffer = []
        try:
            while True:
                # Process all pending actions from queue
                while True:
                    try:
                        # Use timeout to allow checking for interruption
                        timeout = 0.1 if len(action_buffer) == 0 else None
                        actions = action_queue.get(block=len(action_buffer) == 0, timeout=timeout)
                    except queue.Empty:
                        # Timeout occurred, continue to check for shutdown
                        break
                    except KeyboardInterrupt:
                        logging.info("Moz1BrigeInterface server interrupted by user")
                        return
                    for action, timestamp in actions:
                        # Handle special commands
                        if action == "shutdown":
                            logging.info("Moz1BrigeInterface server shutting down")
                            return
                        elif isinstance(action, tuple) and action[0] == "control_event":
                            event_data = action[1]
                            request_id = None
                            sync_call = False

                            # Check if this is a sync control event
                            if isinstance(event_data, tuple) and len(event_data) > 0 and isinstance(event_data[-1], dict) and "_sync" in event_data[-1]:
                                sync_call = True
                                request_id = event_data[-1]["_sync"]["request_id"]
                                event_data = event_data[:-1]  # Remove sync metadata

                            try:
                                result = None
                                if isinstance(event_data, tuple):
                                    event_name = event_data[0]
                                    event_args = event_data[1:]
                                    if hasattr(self._interface, event_name):
                                        logging.info(f"Handling event {event_name} with args {event_args}")
                                        result = getattr(self._interface, event_name)(*event_args)
                                    else:
                                        logger.error(f"Event {event_name} not found in interface")
                                        if sync_call:
                                            response_queue.put({"request_id": request_id, "error": f"Event {event_name} not found"})
                                else:
                                    event_name = event_data
                                    if hasattr(self._interface, event_name):
                                        logging.info(f"Handling event {event_name} with no args")
                                        result = getattr(self._interface, event_name)()
                                    else:
                                        logger.error(f"Event {event_name} not found in interface")
                                        if sync_call:
                                            response_queue.put({"request_id": request_id, "error": f"Event {event_name} not found"})

                                # Send response for sync calls
                                if sync_call:
                                    response_queue.put({"request_id": request_id, "result": result})

                            except Exception as e:
                                logger.error(f"Error handling event {event_name}: {e}")
                                if sync_call:
                                    response_queue.put({"request_id": request_id, "error": str(e)})
                            continue

                        while action_buffer and (timestamp is None or action_buffer[-1][1] >= timestamp):
                            # logger.info(f"timestamp: {timestamp}, action_buffer[-1][1]: {action_buffer[-1][1]}")
                            # logger.info(f"action_buffer: {action_buffer} will pop once")
                            action_buffer.pop()

                        # Buffer regular action with timestamp
                        action_buffer.append((action, timestamp))

                # Execute the most recent action if available
                if action_buffer:
                    action, action_time = action_buffer.pop(0)  # Get most recent
                    if action_time is not None:
                        realtime_precise_sleep(action_time - time.perf_counter())
                    self._interface.send_action(action)
        except KeyboardInterrupt:
            logging.info("Moz1BrigeInterface server interrupted by user")
            return
        except Exception as e:
            logging.error(f"Error in Moz1BrigeInterface server main loop: {e}")
            raise

    @classmethod
    def create_server(cls, rate_hz=120, mu_idxs_list=(None, 0, 1, None), timeout=10.0, virtual=False, simu_mode=False, log_queue=None, enable_logging=False, enable_soft_realtime=False, bind_cpu_idxs=None):
        """
        Create multiprocessing server for Moz1BrigeInterface.

        Returns:
            tuple: (state_queue, action_queue, response_queue, server_process)
        """
        state_queue = mp.Queue(maxsize=1)  # Only keep latest state
        action_queue = mp.Queue()
        response_queue = mp.Queue()  # New response queue for sync operations

        try:
            import rclpy
            rclpy.init()
        except:
            pass

        def _server_process(state_queue, action_queue, response_queue, rate_hz, mu_idxs_list, timeout, virtual, simu_mode, log_queue, enable_logging, enable_soft_realtime, bind_cpu_idxs):
            try:
                server = cls(rate_hz, mu_idxs_list, timeout, virtual, simu_mode, log_queue, enable_logging, enable_soft_realtime, bind_cpu_idxs)
                server.run(state_queue, action_queue, response_queue)
            except Exception as e:
                logging.error(f"Error in Moz1BrigeInterface server process: {e}")
                raise

        server_process = mp.Process(
            target=_server_process,
            args=(state_queue, action_queue, response_queue, rate_hz, mu_idxs_list, timeout, virtual, simu_mode, log_queue, enable_logging, enable_soft_realtime, bind_cpu_idxs)
        )
        server_process.start()

        return state_queue, action_queue, response_queue, server_process


class Moz1BrigeInterfaceStub:
    def __init__(self, rate_hz=120, mu_idxs_list=(None, 0, 1, None), timeout=10.0, virtual=False, simu_mode=False, log_queue=None, enable_logging=False, enable_soft_realtime=False, bind_cpu_idxs=None):
        """
        Client stub for multiprocessing Moz1BrigeInterface.

        Args:
            rate_hz: Control frequency
            mu_idxs_list: Mechanical unit indices
            timeout: Initialization timeout
            virtual: Whether to use virtual robot interface
            simu_mode: Whether to use simulation mode for grippers
            enable_logging: Whether to enable logging below error level (default: False)
            enable_soft_realtime: Enable soft real-time scheduling (default: False)
            bind_cpu_idxs: CPU indices to bind the ROS thread to (default: None)
        """
        self._state_queue, self._action_queue, self._response_queue, self._server_process = \
            _Moz1BrigeInterfaceServer.create_server(rate_hz, mu_idxs_list, timeout, virtual, simu_mode, log_queue, enable_logging, enable_soft_realtime, bind_cpu_idxs)
        
        self._rate_hz = rate_hz
        self._states = None
        self._state_time = None
        self._thread = threading.Thread(target=self._state_receiver_thread)
        self._thread.daemon = True
        self._thread.start()
    
    def _state_receiver_thread(self):
        """Thread to continuously receive robot states."""
        try:
            while True:
                states, state_time = self._state_queue.get()
                self._states = states
                self._state_time = state_time
        except KeyboardInterrupt:
            logging.info("State receiver thread interrupted by user")
            return

    def get_states(self):
        """Get the latest robot states."""
        while self._states is None:
            realtime_precise_sleep(1.0 / self._rate_hz)
        return self._states

    def send_action(self, action: dict, action_timestamp: float = None):
        """
        Send action to robot with optional timestamp.
        Args:
            action: Action dictionary
            action_timestamp: Timestamp for action execution (defaults to current time)
        """
        if action_timestamp is None:
            action_timestamp = time.perf_counter()

        self._action_queue.put([(action, action_timestamp)])
    
    def send_actions(self, actions: list):
        """Send actions to robot with optional timestamp."""
        self._action_queue.put(actions)

    def send_control_event(self, event: str, params = None):
        """Send event to robot (async)."""

        event_data = None
        if params is None:
            event_data = event
        else:
            event_data = (event, params)
        self._action_queue.put([(("control_event", event_data), (time.perf_counter()))])

    def send_control_event_sync(self, event: str, params = None, timeout: float = 5.0):
        """Send event to robot and wait for response (sync).

        Args:
            event: Event name to send
            params: Optional event parameters
            timeout: Timeout in seconds for waiting response

        Returns:
            Response from the event call, or raises exception on error
        """
        import uuid
        import queue as queue_module

        request_id = str(uuid.uuid4())

        # Prepare event data with sync metadata
        if params is None:
            event_data = (event, {"_sync": {"request_id": request_id}})
        else:
            if isinstance(params, tuple):
                event_data = params + ({"_sync": {"request_id": request_id}},)
            else:
                event_data = (event, params, {"_sync": {"request_id": request_id}})

        # Send the sync control event
        self._action_queue.put([(("control_event", event_data), (time.perf_counter()))])

        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._response_queue.get(timeout=0.1)
                if response.get("request_id") == request_id:
                    if "error" in response:
                        raise Exception(f"Control event error: {response['error']}")
                    return response.get("result")
            except queue_module.Empty:
                continue

        raise TimeoutError(f"Timeout waiting for response to event '{event}'")

    def __del__(self):
        """Cleanup when stub is destroyed."""
        try:
            # Signal server to shutdown
            self._action_queue.put([("shutdown", time.perf_counter())])
            # Give server time to shutdown gracefully
            self._server_process.join(timeout=2.0)
            if self._server_process.is_alive():
                self._server_process.terminate()
        except:
            pass

def main(frequency, is_dualarm):
    # according to the robot configuration(dualarm/wholebody moz1), set the mu indexs
    if is_dualarm:
        mu_indexs = (None, 0, 1, None)
    else:
        mu_indexs = (0, 1, 2, None)

    try:
        rclpy.init()
    except:
        pass
    print("rclpy init finish")

    try:
        moz1_server = Moz1BrigeInterface(rate_hz=frequency, mu_idxs_list=mu_indexs)
        print("Robot is online and ready")

        time.sleep(2)
        states = moz1_server.get_states()
        print("Current states:", states)

    except RuntimeError as e:
        print(f"Failed to initialize robot: {e}")
        return
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    is_dualarm = True
    frequency = 30
    main(frequency, is_dualarm)
