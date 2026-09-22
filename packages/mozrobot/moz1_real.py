import logging
from typing import List, Optional, Dict, Any  # noqa: UP035
from dataclasses import dataclass

import torch
import numpy as np

from mozrobot.envs import real_env_moz1 as _real_env
from mozrobot.configs import MOZ1RobotConfig
from mozrobot.spirit_robot_base import SpiritRobotBase
from mozrobot.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from mozrobot.step_control.utils import make_step_axis_controller_from_config
from mozrobot.process_manager import get_process_manager

def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


@dataclass
class JointState:
    """Store smoothed state for a single joint"""
    last_position: float = 0.0
    last_time: float = 0.0
    is_first_call: bool = True

class MOZ1Robot(SpiritRobotBase):
    _K_MAX_VELOCITY = 0.5
    _K_SMOOTH_FACTOR = 0.3
    _EPSILON = 0.0001
    _APPROACH_TOLERANCE = 0.05

    _ARM_JOINTS_PER_ARM = 7
    _TOTAL_ARM_JOINTS = _ARM_JOINTS_PER_ARM * 2

    def __init__(
        self,
        config: MOZ1RobotConfig,
    ) -> None:
        self.robot_structure = config.structure

        if config.structure == "dualarm":
            enabled_mechunits = {"leftarm": 7, "rightarm": 7}
        elif config.structure == "wholebody_without_base":
            enabled_mechunits = {"leftarm": 7, "rightarm": 7, "torso": 6}
        elif config.structure == "wholebody":
            enabled_mechunits = {"leftarm": 7, "rightarm": 7, "torso": 6, "base": 1}
        else:
            raise ValueError(f"Invalid structure: {config.structure}")

        super().__init__(config, enabled_mechunits=enabled_mechunits, enabled_end_effectors={"leftarm": ("gripper", 1), "rightarm": ("gripper", 1)})
        self.robot_type = self.config.type
        self.is_connected = False
        self.cameras = [None]*3
        self.process_manager = get_process_manager()

    def connect(self) -> None:
        # Create multiprocess log handler for child processes
        from mozrobot.utils import create_multiprocess_log_handler
        self.log_queue, self.log_listener = create_multiprocess_log_handler()
        self._env = _real_env.make_real_env(realsense_serials=self.config.realsense_serials.strip().split(","),
                                            camera_resolutions=self.camera_resolution,
                                            structure=self.config.structure,
                                            robot_control_hz=self.config.robot_control_hz,
                                            simu_mode=self.config.simu_mode,
                                            virtual_robot=self.config.virtual_robot,
                                            log_queue=self.log_queue,
                                            enable_logging=self.config.enable_driver_logging,
                                            no_camera=self.config.no_camera,
                                            enable_soft_realtime=self.config.enable_soft_realtime,
                                            bind_cpu_idxs=self.config.bind_cpu_idxs,
                                            disabled_cameras=self.config.disabled_cameras)
        self.is_connected = True

        # Register processes with the process manager for graceful shutdown
        if hasattr(self._env, 'arm_controller') and hasattr(self._env.arm_controller, '_server_process'):
            self.process_manager.register_process(self._env.arm_controller._server_process)
        if hasattr(self._env, 'arm_controller') and hasattr(self._env.arm_controller, '_action_queue'):
            self.process_manager.register_queue(self._env.arm_controller._action_queue)
        if hasattr(self._env, 'image_recorder'):
            for camera_name, camera_process in getattr(self._env.image_recorder, '_camera_processes', {}).items():
                if camera_process and hasattr(camera_process, 'process'):
                    self.process_manager.register_process(camera_process.process)

        if self.config.teleop_device is not None and hasattr(self.config.teleop_device, "enable_moz_teleop") and self.config.teleop_device.enable_moz_teleop:
            logging.info(f"Connecting to teleop device: {self.config.teleop_device}")
            try:
                from mozrobot.teleoperations.moz_teleop_adapter import MozTeleopAdapter
                self._teleop_adapter = MozTeleopAdapter(self.config.teleop_device, list(self.enabled_mechunits.keys()), self.robot_structure)

                is_connected = self._teleop_adapter.connect_device()
                if not is_connected:
                    raise RuntimeError("Failed to connect to teleop device.")

                self._teleop_device_enabled = True
            except Exception as e:
                logging.error(f"Failed to connect to teleop device: {e}")
                raise RuntimeError(f"Failed to connect to teleop device: {e}")

            logging.info(f"Teleop device connected: {self._teleop_device_enabled}")
            if not has_method(self._teleop_adapter, "get_teleop_cmd"):
                logging.error("get_teleop_cmd not implemented.")
                raise RuntimeError("get_teleop_cmd not implemented.")

        # connect StepAxisController
        if self.config.step_control_device:
            try:
                self._step_axis_controller = make_step_axis_controller_from_config(self.config.step_control_device)
                if not self._step_axis_controller.connect():
                    logging.error("Step axis controller connect failed.")
                    self._step_axis_controller = None
            except Exception as e:
                logging.error(f"init step axis controller failed: {e}")
                self._step_axis_controller = None

        self.reset()

    def disconnect(self) -> None:
        try:
            logging.info("Starting MOZ1Robot graceful shutdown...")

            # Use ProcessManager for coordinated shutdown
            self.process_manager.shutdown_all()

            # Properly shutdown the environment and its subprocesses
            if hasattr(self, '_env') and self._env is not None:
                # Shutdown arm controller server process
                if hasattr(self._env, 'arm_controller') and self._env.arm_controller is not None:
                    try:
                        # Send shutdown command to server process
                        self._env.arm_controller._action_queue.put([("shutdown", 0)])
                        # Give it some time to shutdown gracefully
                        import time
                        time.sleep(0.5)
                        # Terminate the server process if it's still running
                        if hasattr(self._env.arm_controller, '_server_process') and self._env.arm_controller._server_process.is_alive():
                            self._env.arm_controller._server_process.terminate()
                            self._env.arm_controller._server_process.join(timeout=2.0)
                            if self._env.arm_controller._server_process.is_alive():
                                self._env.arm_controller._server_process.kill()
                    except Exception as e:
                        logging.warning(f"Error shutting down arm controller: {e}")

                # Shutdown image recorder processes
                if hasattr(self._env, 'image_recorder') and self._env.image_recorder is not None:
                    try:
                        self._env.image_recorder.stop()
                    except Exception as e:
                        logging.warning(f"Error shutting down image recorder: {e}")

            # disconnect StepAxisController
            if hasattr(self, "_step_axis_controller") and self._step_axis_controller is not None:
                self._step_axis_controller.disconnect()

            # Stop log listener if it exists
            if hasattr(self, 'log_listener') and self.log_listener is not None:
                try:
                    self.log_listener.stop()
                except Exception as e:
                    logging.warning(f"Error stopping log listener: {e}")

        except Exception as e:
            logging.error(f"Error during disconnect: {e}")
        finally:
            self._env = None
            self.is_connected = False
            logging.info("MOZ1Robot disconnect completed")

    def start_teleop(self, is_dagger_mode=False):
        logging.info(f"start teleop, is dagger mode: {is_dagger_mode}")

        if self._teleop_adapter is None:
            raise RuntimeError("Teleop operation not connected.")

        is_success = self._teleop_adapter.start_teleop(is_dagger_mode=is_dagger_mode)
        if not is_success:
            raise RuntimeError("Failed to start teleop.")

    def stop_teleop(self) -> None:
        if self._teleop_adapter is None:
            logging.warning("Teleop operation not connected, skipping stop teleop")
            return

        self._teleop_adapter.stop_teleop()
    
    def teleop_step(self, record_data:bool = False, record_image: bool = False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        if self._teleop_adapter is None:
            raise RuntimeError("Teleop operation not connected.")

        teleop_action = self._teleop_adapter.get_teleop_cmd()
        if teleop_action is None:
            raise RuntimeError("Teleop operation has no valid action data.")

        logging.debug(f"teleop_actions: {teleop_action}")

        if not record_data:
            return None, None

        observation = self.capture_robot_observation()
        if record_image:
            observation.update(self.capture_images())

        # combine teleop_action using current feedback
        # Due to the accumulation of errors in force control mode (especially torso), do not use current state as the control command
        if "torso" in self.enabled_mechunits and "torso_cmd_cart_pos" not in teleop_action:
            teleop_action["torso_cmd_cart_pos"] = np.asarray(observation["torso_state_cart_pos"], dtype=np.float32)
        if "leftarm" in self.enabled_mechunits and "leftarm_cmd_cart_pos" not in teleop_action:
            teleop_action["leftarm_cmd_cart_pos"] = np.asarray(observation["leftarm_state_cart_pos"], dtype=np.float32)
        if "rightarm" in self.enabled_mechunits and "rightarm_cmd_cart_pos" not in teleop_action:
            teleop_action["rightarm_cmd_cart_pos"] = np.asarray(observation["rightarm_state_cart_pos"], dtype=np.float32)

        if "base" in self.enabled_mechunits and "base_cmd_speed" not in teleop_action:
            teleop_action["base_cmd_speed"] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

        return observation, teleop_action

    def reset_robot_positions(self, left_arm_joints: Optional[list] = None,
                            right_arm_joints: Optional[list] = None,
                            torso_joints: Optional[list] = None,
                            gripper_positions: Optional[list] = None) -> bool:
        if self._env is None:
            raise RobotDeviceNotConnectedError(
                "Moz1 robot is not connected. You need to run `robot.connect()`."
            )
        self._env.update_reset_position(left_arm_joints, right_arm_joints, torso_joints, gripper_positions)
        return self._env.move_robot_to_reset_position()

    def enable_external_following_mode(self):
        if self._env is None:
            raise RobotDeviceNotConnectedError(
                "Moz1 robot is not connected. You need to run `robot.connect()`."
            )
        self._env.enable_external_following_mode()

    def reset(self):
        logging.info("Moz1 robot reset")
        self._env.reset()
        if self.config.step_control_device:
            self._step_axis_controller.home_motor()

    def support_soft_real_time(self) -> bool:
        return True

    @property
    def teleop_device_enabled(self) -> bool:
        return self._teleop_device_enabled

    def get_teleop_device_info(self) -> dict:
        """获取遥操作设备信息"""
        if self._teleop_adapter is None:
            return {"error": "Teleop adapter not connected"}

        try:
            return self._teleop_adapter.get_device_connection_info()
        except Exception as e:
            return {"error": f"Failed to get device info: {e}"}

    @property
    def control_hz(self) -> int:
        return self.config.robot_control_hz

    @property
    def observation_hz(self) -> int:
        return self.config.robot_observation_hz

    @property
    def is_robot_connected(self) -> bool:
        return self.is_connected