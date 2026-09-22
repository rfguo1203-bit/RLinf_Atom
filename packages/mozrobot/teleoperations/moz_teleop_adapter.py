"""
MozTeleop API自适应适配器实现

本模块提供了与moz_teleop系统集成的统一适配器类，用于替换原有的直接硬件访问方式。
适配器类通过ROS服务和话题与moz_teleop系统通信，并根据teleop_state自动适应VR或外骨骼设备。
"""

from ast import Raise
import json
import logging
from re import L
import threading
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import scipy.spatial.transform as st

try:
    import rclpy
    import rclpy.context
    from rclpy.node import Node
    from rclpy.client import Client
    from rclpy.subscription import Subscription
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String, Float64MultiArray
    from geometry_msgs.msg import Vector3, Pose, Point, Quaternion

    from mc_core_interface.msg import MechUnitCmdArray, MechUnitCmd, MechUnitState, MechUnitStateArray
    from mc_core_interface.srv import RobotCmdService

    ROS_AVAILABLE = True
except ImportError:
    logging.warning("ROS2 not available. MozTeleop adapter will not work without ROS2.")
    ROS_AVAILABLE = False

# MOZ1机器人结构映射 (torso, leftarm, rightarm, base)
MOZ1_STRUCTURE_MAPPING = {
    "dualarm": (None, 0, 1, None),
    "wholebody_without_base": (0, 1, 2, None),
    "wholebody": (1, 2, 3, 0)
}
class MozTeleopServiceClient:
    """moz_teleop ROS服务客户端封装类"""

    def __init__(self, node: Node, service_name: str, timeout: float = 5.0):
        self.node = node
        self.service_name = service_name
        self.timeout = timeout
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """设置ROS服务客户端"""
        self.client = self.node.create_client(RobotCmdService, self.service_name)
        logging.info(f"Created RobotCmdService client for {self.service_name}")

    def call_service(self, cmd: str, data: str = "") -> Tuple[bool, str]:
        """
        调用moz_teleop服务

        Args:
            cmd: 命令名称
            data: 命令参数（JSON格式）

        Returns:
            (success, response_data): 成功标志和响应数据
        """
        if not self.client:
            logging.error(f"Service client not available for {self.service_name}")
            return False, "Service client not available"

        if not self.client.wait_for_service(timeout_sec=self.timeout):
            logging.error(f"Service {self.service_name} not available within {self.timeout}s")
            return False, f"Service timeout after {self.timeout}s"

        try:
            request = RobotCmdService.Request()
            request.cmd = cmd
            request.data = data

            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout)

            if future.result() is not None:
                response = future.result()
                return response.success, response.data
            else:
                logging.error(f"Service call failed for {cmd}")
                return False, "Service call failed"

        except Exception as e:
            logging.error(f"Error calling service {self.service_name}: {e}")
            return False, str(e)

class RosDataReaderThread(threading.Thread):
    """异步ROS数据读取线程"""

    def __init__(self, config, shared_data: dict, data_lock: threading.RLock, 
                 data_condition: threading.Condition, shutdown_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.shared_data = shared_data
        self.data_lock = data_lock
        self.data_condition = data_condition
        self.shutdown_event = shutdown_event
        self.logger = logging.getLogger("RosDataReaderThread")
        self.node = None

    def run(self):
        """线程主函数"""
        try:
            if not ROS_AVAILABLE:
                self.logger.error("ROS2 not available in data reader thread")
                return

            self.logger.info("Creating ROS node in thread...")

            try:
                # 在线程中创建ROS节点，复用主进程的ROS上下文
                self.node = Node(f"moz_teleop_data_reader_{int(time.time())}")
                self.executor = rclpy.executors.SingleThreadedExecutor()
                self.executor.add_node(self.node)
            except Exception as e:
                self.logger.error(f"Error creating ROS node: {e}")
                return

            # 数据缓存
            latest_data = {
                'teleop_state': None,
                'mix_commands': None,
                'gripper_commands': None,
                'base_commands': None,
                'timestamp': time.time()
            }

            # 设置订阅
            self._setup_subscribers(self.node, latest_data)

            self.logger.info("ROS data reader thread started")

            # 主循环
            while not self.shutdown_event.is_set() and rclpy.ok():
                try:
                    # 执行回调
                    try:
                        self.executor.spin_once(timeout_sec=0.001)
                    except Exception as e:
                        self.logger.error(f"Error spinning ROS node: {e}")
                        break

                    # 通过队列发送数据更新
                    data_copy = latest_data.copy()
                    data_copy['timestamp'] = time.time()

                    # 直接更新共享数据（只保存最新数据）
                    with self.data_condition:
                        self.shared_data.update(data_copy)
                        self.data_condition.notify()  # 通知主线程有新数据

                except Exception as e:
                    if not self.shutdown_event.is_set():
                        self.logger.error(f"Error in ROS data reader loop: {e}")
                    break

            # 清理
            self.logger.info("ROS data reader thread stopped")
            if self.executor:
                self.executor.shutdown()
                self.executor = None

            if self.node:
                self.node.destroy_node()
                self.node = None

        except Exception as e:
            self.logger.error(f"Fatal error in ROS data reader thread: {e}")
        finally:
            # 确保清理节点
            if self.node:
                try:
                    self.node.destroy_node()
                except Exception as e:
                    self.logger.debug(f"Error destroying ROS node: {e}")

    def _setup_subscribers(self, node: Node, latest_data: dict):
        """设置ROS话题订阅"""
        subscribers = {}

        try:
            # 订阅遥操作状态
            subscribers['teleop_state'] = node.create_subscription(
                String,
                self.config.teleop_state_topic,
                lambda msg: self._update_data(latest_data, 'teleop_state', self._parse_teleop_state(msg)),
                5
            )

            subscribers['mix_commands'] = node.create_subscription(
                MechUnitCmdArray,
                self.config.mix_command_topic,
                lambda msg: self._update_data(latest_data, 'mix_commands', msg),
                5
            )
            logging.info("Successfully subscribed to MechUnitCmdArray topic")

            # 订阅夹爪指令
            subscribers['gripper_commands'] = node.create_subscription(
                Float64MultiArray,
                self.config.gripper_command_topic,
                lambda msg: self._update_data(latest_data, 'gripper_commands', list(msg.data)),
                5
            )

            # 订阅底盘指令
            subscribers['base_commands'] = node.create_subscription(
                Vector3,
                getattr(self.config, 'base_command_topic', 'mx_base_vel_command'),
                lambda msg: self._update_data(latest_data, 'base_commands', self._parse_base_command(msg)),
                5
            )

            self.logger.info(f"Set up {len(subscribers)} ROS subscribers")
            return subscribers

        except Exception as e:
            logging.error(f"Failed to set up ROS subscribers: {e}")
            return {}

    def _update_data(self, latest_data: dict, key: str, value):
        """更新数据缓存"""
        latest_data[key] = value

    def _parse_teleop_state(self, msg: String) -> dict:
        """解析遥操作状态消息"""
        try:
            return json.loads(msg.data)
        except Exception as e:
            logging.error(f"Error parsing teleop state: {e}")
            return {}

    def _parse_base_command(self, msg: Vector3) -> List[float]:
        """解析底盘指令消息"""
        return [msg.x, msg.y, msg.z]
class MozTeleopAdapter:
    """
    moz_teleop API的统一自适应适配器类

    根据teleop_state topic自动检测设备类型（VR或HRPI外骨骼），
    并相应地处理控制指令。使用异步进程确保ROS数据读取的实时性。
    """

    def __init__(self, config, enabled_mechunits: List[str], robot_structure: str = "dualarm"):
        """
        初始化自适应适配器

        Args:
            config: 遥操作配置对象
            enabled_mechunits: 启用的机械单元列表
            robot_structure: 机器人结构类型
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS2 is required for MozTeleop adapter but not available")

        self.config = config
        self.enabled_mechunits = enabled_mechunits
        self.robot_structure = robot_structure

        # 从MOZ1_STRUCTURE_MAPPING获取mu_idx映射
        if robot_structure not in MOZ1_STRUCTURE_MAPPING:
            raise ValueError(f"Invalid robot structure: {robot_structure}")

        self.mu_idx_mapping = MOZ1_STRUCTURE_MAPPING[robot_structure]
        logging.info(f"Using mu_idx mapping for {robot_structure}: {self.mu_idx_mapping}")

        # ROS相关
        self.node = None
        self.service_client = None

        # 线程共享数据存储 - 只保存最新数据
        self._latest_shared_data = {
            'teleop_state': None,
            'mix_commands': None,
            'gripper_commands': None,
            'base_commands': None,
            'timestamp': 0
        }
        self._data_lock = threading.RLock()  # 使用可重入锁
        self._data_condition = threading.Condition(self._data_lock)  # 用于通知数据更新
        self._shutdown_event = threading.Event()
        self._data_reader_thread = None

        # 本地数据缓存（用于向外部提供数据）
        self._latest_data = {
            'teleop_state': None,
            'mix_commands': None,
            'gripper_commands': None,
            'base_commands': None,
            'timestamp': 0
        }

        # 遥操作信息
        self._teleop_info = {
            "device_type": None,
            "device_state": None,
            "last_error_info": None,
            "is_connected": False,
            "is_teleop_active": False
        }

        logging.info(f"Initialized MozTeleopAdapter with structure {robot_structure}")

    def connect_device(self) -> bool:
        """连接到moz_teleop服务"""
        logging.info(f"Connecting to moz_teleop service...")

        try:
            # 初始化ROS（主进程中的服务客户端）
            if not rclpy.ok():
                rclpy.init()

            # 创建ROS节点用于服务调用
            self.node = Node(f"moz_teleop_adapter_client_{int(time.time())}")

            # 创建服务客户端
            self.service_client = MozTeleopServiceClient(
                self.node,
                self.config.ros_service_name,
                self.config.ros_service_timeout
            )

            # 启动异步数据读取线程
            self._start_data_reader_thread()

            # 检查moz_teleop系统信息
            success, system_info = self.service_client.call_service("GetTeleopSystemInfo")
            if success:
                logging.info(f"Connected to moz_teleop system: {system_info}")
                self._teleop_info["device_type"] = json.loads(system_info).get("teleop_device_type")
            else:
                logging.error(f"Failed to connect to moz_teleop")
                return False

            # 等待并检测设备连接
            if not self._waif_for_teleop_device_connected():
                logging.error("Failed to wait teleop device connected")
                return False

            self._teleop_info["is_connected"] = True
            return True

        except Exception as e:
            logging.error(f"Failed to connect to moz_teleop: {e}")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        """断开与moz_teleop的连接"""
        logging.info("Disconnecting from moz_teleop...")

        try:
            # 停止遥操作
            if self._teleop_info["is_teleop_active"]:
                self.stop_teleop()

            # 停止数据读取线程
            self._stop_data_reader_thread()

            # 清理ROS资源
            if self.node:
                self.node.destroy_node()
                self.node = None

            self._teleop_info["is_connected"] = False
            logging.info("Disconnected from moz_teleop")

        except Exception as e:
            logging.error(f"Error during disconnect: {e}")

    def start_teleop(self, is_dagger_mode: bool = False) -> bool:
        """启动遥操作设备"""
        if not self._teleop_info["is_connected"]:
            logging.error("Device not connected. Call connect_device() first.")
            return False

        logging.info("Starting moz_teleop device...")

        if self._teleop_info["is_teleop_active"]:
            logging.warning("Teleop device is already active")
            return True

        try:
            control_params = json.dumps(self._get_adaptive_control_params(is_dagger_mode))
            success, response = self.service_client.call_service(
                "StartTeleop",
                control_params
            )

            if not success:
                logging.error(f"Could not start teleop")
                return False

            self._teleop_info["is_teleop_active"] = self.wait_for_teleop_device_active()
            if not self._teleop_info["is_teleop_active"]:
                logging.error("Failed to wait for teleop device active")
                return False

        except Exception as e:
            logging.error(f"Failed to start moz_teleop device: {e}")
            return False
        
        return True

    def stop_teleop(self) -> None:
        """停止遥操作"""
        if not self._teleop_info["is_connected"] or not self._teleop_info["is_teleop_active"]:
            return

        logging.info("Stopping moz_teleop...")

        try:
            success, response = self.service_client.call_service("StopTeleop")
            if not success:
                logging.warning(f"Could not stop teleop cleanly: {response}")

            self._teleop_info["is_teleop_active"] = False

            # 清理数据缓存
            with self._data_lock:
                self._latest_data = {
                    'teleop_state': None,
                    'mix_commands': None,
                    'gripper_commands': None,
                    'base_commands': None,
                    'timestamp': 0
                }

            logging.info("MozTeleop stopped")

        except Exception as e:
            logging.error(f"Error stopping moz_teleop: {e}")

    def get_teleop_cmd(self) -> Dict[str, Any]:
        """
        获取遥操作指令，自动适应VR或外骨骼设备
        支持部分topic可用的情况，只处理有效的topic数据

        Returns:
            teleop_commands: 遥操作指令字典
        """
        if not self._is_data_reader_thread_health():
            logging.error("Data reader thread is not healthy")
            raise RuntimeError("Data reader thread is not healthy")

        # 更新数据缓存
        self._update_data_from_shared()

        self._update_teleop_state(self._latest_data.get('teleop_state', {}))

        if not self._is_teleop_device_state_ok():
            logging.error("Device not ready for teleop commands")
            raise RuntimeError("Device not ready for teleop commands")

        # 获取最新数据
        with self._data_lock:
            mix_commands = self._latest_data.get('mix_commands')
            gripper_commands = self._latest_data.get('gripper_commands')
            base_commands = self._latest_data.get('base_commands')

        # 初始化teleop_cmd，设置默认值
        teleop_cmd = self._get_default_teleop_cmd()

        try:
            # 处理机械单元指令（如果可用）
            if mix_commands is not None:
                self._process_mechanical_unit_commands(teleop_cmd, mix_commands)
            else:
                logging.debug("No mix commands available, using default arm positions")

            # 处理夹爪指令（如果可用）
            if gripper_commands is not None:
                self._process_gripper_commands(teleop_cmd, gripper_commands)
            else:
                logging.debug("No gripper commands available, using default gripper positions")

            # 处理底盘指令（如果可用且启用）
            if base_commands is not None:
                teleop_cmd["base_cmd_speed"] = np.asarray(base_commands, dtype=np.float32)

        except Exception as e:
            logging.error(f"Error processing teleop commands: {e}")
            return self._get_default_teleop_cmd()

        return teleop_cmd

    def _start_data_reader_thread(self):
        """启动异步数据读取线程"""
        self._shutdown_event.clear()

        self._data_reader_thread = RosDataReaderThread(
            self.config, self._latest_shared_data, self._data_lock, 
            self._data_condition, self._shutdown_event
        )
        self._data_reader_thread.start()
        logging.info("Started ROS data reader thread")

    def _stop_data_reader_thread(self):
        """停止异步数据读取线程"""
        if self._data_reader_thread and self._data_reader_thread.is_alive():
            self._shutdown_event.set()
            self._data_reader_thread.join(timeout=3.0)
            if self._data_reader_thread.is_alive():
                logging.warning("Data reader thread did not terminate gracefully")

        # 清空共享数据
        with self._data_condition:
            self._latest_shared_data.update({
                'teleop_state': None,
                'mix_commands': None,
                'gripper_commands': None,
                'base_commands': None,
                'timestamp': 0
            })

    def _update_data_from_shared(self):
        """从共享数据更新本地缓存"""
        try:
            # 从共享数据复制最新数据到本地缓存
            with self._data_lock:
                # 检查是否有新数据（通过时间戳判断）
                if (self._latest_shared_data['timestamp'] > self._latest_data['timestamp']):
                    self._latest_data.update(self._latest_shared_data.copy())
                    return True
            return False
        except Exception as e:
            logging.debug(f"Error updating data from shared storage: {e}")
            return False

    def _update_teleop_state(self, teleop_state: dict):
        assert teleop_state, "teleop_state is Empty"

        device_info = teleop_state.get("device_info", {})
        if device_info:
            self._teleop_info["device_state"] = device_info.get("device_state")
            if self._teleop_info["device_state"] != "connect":
                self._teleop_info["last_error_info"] = device_info.get("error_info")

        self._teleop_info["is_teleop_active"] = teleop_state.get("is_teleop")

    def _restart_data_reader_thread(self):
        """重启数据读取线程"""
        try:
            logging.info("Restarting data reader thread...")
            self._stop_data_reader_thread()
            time.sleep(0.5)  # 短暂等待确保线程完全停止
            self._start_data_reader_thread()
            logging.info("Data reader thread restarted successfully")
        except Exception as e:
            logging.error(f"Failed to restart data reader thread: {e}")

    def _waif_for_teleop_device_connected(self, timeout: float = 3.0) -> bool:
        """等待设备类型检测完成"""
        start_time = time.time()

        self._teleop_info["device_state"] = None
        while time.time() - start_time < timeout and self._teleop_info["device_state"] is None:
            # 更新数据缓存
            self._update_data_from_shared()
            # 检查teleop_state中的设备信息
            with self._data_lock:
                teleop_state = self._latest_data.get('teleop_state')

            if teleop_state:
                device_info = teleop_state.get("device_info", {})
                device_type = device_info.get("device_type")
                device_state = device_info.get("device_state")

                if device_type != self._teleop_info["device_type"]:
                    logging.error("Device type is not the same as the detected device type")
                    return False

                if device_state.lower() != "connect":
                    self._teleop_info["last_error_info"] = device_info.get("error_info")
                    logging.error(f"Device state is not connect, error info: {self._teleop_info['last_error_info']}")
                    return False

                self._teleop_info["device_state"] = device_state
                return True

            time.sleep(0.1)

        logging.error("Failed to wait for teleop device connected")
        return False

    def wait_for_teleop_device_active(self, timeout: float = 3.0) -> bool:
        """等待设备激活"""
        start_time = time.time()

        is_teleop_active = False
        while time.time() - start_time < timeout and not is_teleop_active:
            self._update_data_from_shared()
            with self._data_lock:
                teleop_state = self._latest_data.get('teleop_state')

            if teleop_state:
                is_teleop_active = teleop_state.get("is_teleop")
                if is_teleop_active:
                    return True

            time.sleep(0.1)

        logging.error("Failed to wait for teleop device active")
        return False

    def _get_adaptive_control_params(self, is_dagger_mode: bool = False) -> dict:
        """根据检测到的设备类型和控制模式获取自适应控制参数"""
        if self._teleop_info["device_type"] == "HRPI":
            # 外骨骼设备：根据控制模式决定参数
            if is_dagger_mode:
                # Dagger模式：relative + cartesian
                logging.info("Using HRPI Dagger mode: relative + cartesian")
                return {
                    "teleop_control_mode": "relative",
                    "teleop_control_space": "cartesian"
                }
            else:
                # 普通采集模式：absolute + joint
                logging.info("Using HRPI collection mode: absolute + joint")
                return {
                    "teleop_control_mode": "absolute",
                    "teleop_control_space": "joint"
                }
        else:
            # VR设备：relative + cartesian
            logging.info("Using VR mode: relative + cartesian")
            return {
                "teleop_control_mode": "relative",
                "teleop_control_space": "cartesian"
            }

    def _process_gripper_commands(self, teleop_cmd: dict, gripper_commands: List):
        """处理夹爪指令（仅在数据可用时调用）"""
        if len(gripper_commands) >= 2:
            teleop_cmd["leftarm_gripper_cmd_pos"] = np.asarray([gripper_commands[0]], dtype=np.float32)
            teleop_cmd["rightarm_gripper_cmd_pos"] = np.asarray([gripper_commands[1]], dtype=np.float32)
        elif len(gripper_commands) == 1:
            # 如果只有一个夹爪数据，应用到左臂
            teleop_cmd["leftarm_gripper_cmd_pos"] = np.asarray([gripper_commands[0]], dtype=np.float32)
            logging.debug("Only one gripper command available, applied to left arm")
        else:
            logging.debug("Empty gripper commands received")

    def _process_mechanical_unit_commands(self, teleop_cmd: dict, mix_commands):
        """处理机械单元指令，根据mu_idx映射和设备类型自适应"""
        if not hasattr(mix_commands, 'cmds'):
            return

        torso_idx, leftarm_idx, rightarm_idx, base_idx = self.mu_idx_mapping

        for command in mix_commands.cmds:
            mu_idx = command.mu_idx

            # 左臂控制
            if mu_idx == leftarm_idx and "leftarm" in self.enabled_mechunits:
                self._process_arm_command(teleop_cmd, command, "leftarm")

            # 右臂控制
            elif mu_idx == rightarm_idx and "rightarm" in self.enabled_mechunits:
                self._process_arm_command(teleop_cmd, command, "rightarm")

            # 躯干控制
            elif mu_idx == torso_idx and "torso" in self.enabled_mechunits:
                self._process_torso_command(teleop_cmd, command)

    def _process_arm_command(self, teleop_cmd: dict, command, arm_name: str):
        """处理单臂指令"""
        if command.use_jnt:  # 关节控制（主要用于外骨骼）
            if len(command.jnt_pos) >= 7:
                teleop_cmd[f"{arm_name}_cmd_joint_pos"] = np.asarray(command.jnt_pos[:7], dtype=np.float32)
        else:  # 笛卡尔控制（主要用于VR）
            pose = command.end_pose
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            quat_xyzw = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
            cart_pos = np.concatenate([pos, rot_vec], dtype=np.float32)
            teleop_cmd[f"{arm_name}_cmd_cart_pos"] = cart_pos

    def _process_torso_command(self, teleop_cmd: dict, command):
        """处理躯干指令"""
        # 躯干目前只能使用笛卡尔控制
        if not command.use_jnt:  
            pose = command.end_pose
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            quat_xyzw = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
            cart_pos = np.concatenate([pos, rot_vec], dtype=np.float32)
            teleop_cmd["torso_cmd_cart_pos"] = cart_pos

    def _get_default_teleop_cmd(self) -> dict:
        """获取默认的遥操作指令"""
        default_cmd = {
            "leftarm_gripper_cmd_pos": np.asarray([0.0], dtype=np.float32),
            "rightarm_gripper_cmd_pos": np.asarray([0.0], dtype=np.float32)
        }
        return default_cmd

    def _is_teleop_device_state_ok(self) -> bool:
        if self._teleop_info["device_state"] != "connect":
            logging.error(f"Teleop device state is not connect, error info: {self._teleop_info['last_error_info']}")
            return False

        if not self._teleop_info["is_teleop_active"]:
            logging.error("Teleop not active")
            return False

        return True 

    def _is_data_reader_thread_health(self):
        """检查数据读取线程健康状态"""
        if self._data_reader_thread is None:
            return False

        if not self._data_reader_thread.is_alive():
            logging.error("Data reader thread has died, attempting to restart...")
            self._restart_data_reader_thread()
            return False

        return True

    def get_device_connection_info(self) -> dict:
        """
        获取设备连接信息

        Returns:
            dict: 包含设备连接状态和错误信息的字典
        """
        return self._teleop_info