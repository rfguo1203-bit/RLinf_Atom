# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MOZ SDK connection with one locally owned command stream."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from rlinf.robotics.parts.base import Action, Features, Observation, RobotPart
from rlinf.utils.logging import get_logger


@dataclass
class MOZSnapshot:
    """One MOZ state and image reading used by a policy step."""

    right_tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    right_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_gripper: float = 0.0
    left_tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    left_gripper: float = 0.0
    torso_tcp_pose: Optional[np.ndarray] = None
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


class _DummyMOZRobot:
    """Small in-process MOZ substitute for schema and environment tests."""

    def __init__(self, image_size: tuple[int, int]) -> None:
        height, width = image_size
        self.right_tcp_pose = np.zeros(6, dtype=np.float32)
        self.right_joint_positions = np.zeros(7, dtype=np.float32)
        self.right_gripper = 0.0
        self.left_tcp_pose = np.zeros(6, dtype=np.float32)
        self.left_gripper = 0.0
        self.torso_tcp_pose = np.zeros(6, dtype=np.float32)
        self.frames = {
            "cam_high": np.zeros((height, width, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((height, width, 3), dtype=np.uint8),
        }
        self.external_following = False
        self.teleop_started = False

    def connect(self, **_: Any) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def enable_external_following_mode(self) -> None:
        self.external_following = True

    def disable_external_following_mode(self) -> None:
        self.external_following = False

    def reset_robot_positions(self, **_: Any) -> bool:
        return True

    def send_action(self, action: Mapping[str, Any], action_time: Any = None) -> None:
        del action_time
        if "rightarm_cmd_cart_pos" in action:
            self.right_tcp_pose = np.asarray(
                action["rightarm_cmd_cart_pos"], dtype=np.float32
            ).reshape(6)
        if "rightarm_gripper_cmd_pos" in action:
            self.right_gripper = float(
                np.asarray(action["rightarm_gripper_cmd_pos"], dtype=np.float32).reshape(-1)[0]
            )
        if "leftarm_cmd_cart_pos" in action:
            self.left_tcp_pose = np.asarray(
                action["leftarm_cmd_cart_pos"], dtype=np.float32
            ).reshape(6)
        if "leftarm_gripper_cmd_pos" in action:
            self.left_gripper = float(
                np.asarray(action["leftarm_gripper_cmd_pos"], dtype=np.float32).reshape(-1)[0]
            )
        if "torso_cmd_cart_pos" in action:
            self.torso_tcp_pose = np.asarray(
                action["torso_cmd_cart_pos"], dtype=np.float32
            ).reshape(6)

    def capture_robot_observation(self) -> dict[str, np.ndarray]:
        return {
            "rightarm_state_cart_pos": self.right_tcp_pose.copy(),
            "rightarm_state_joint_pos": self.right_joint_positions.copy(),
            "rightarm_gripper_state_pos": np.asarray([self.right_gripper], dtype=np.float32),
            "leftarm_state_cart_pos": self.left_tcp_pose.copy(),
            "leftarm_gripper_state_pos": np.asarray([self.left_gripper], dtype=np.float32),
            "torso_state_cart_pos": self.torso_tcp_pose.copy(),
        }

    def capture_images(self) -> dict[str, np.ndarray]:
        return {name: frame.copy() for name, frame in self.frames.items()}

    def start_teleop(self, is_dagger_mode: bool = False) -> None:
        del is_dagger_mode
        self.teleop_started = True

    def stop_teleop(self) -> None:
        self.teleop_started = False

    def teleop_step(
        self, record_data: bool = True, record_image: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        observation = self.capture_robot_observation()
        if record_image:
            observation.update(self.capture_images())
        return observation, self._hold_command()

    def _hold_command(self) -> dict[str, np.ndarray]:
        return {
            "leftarm_cmd_cart_pos": self.left_tcp_pose.copy(),
            "leftarm_gripper_cmd_pos": np.asarray([self.left_gripper], dtype=np.float32),
            "rightarm_cmd_cart_pos": self.right_tcp_pose.copy(),
            "rightarm_gripper_cmd_pos": np.asarray([self.right_gripper], dtype=np.float32),
            "torso_cmd_cart_pos": self.torso_tcp_pose.copy(),
        }


class MOZConnection(RobotPart):
    """Own one MOZ SDK session and serialize all controller writes.

    The environment submits complete, already validated targets at its policy
    rate. This connection holds the freshest target at the robot control rate,
    so a Ray transport delay never becomes a second command writer.
    """

    def __init__(
        self,
        *,
        structure: str = "wholebody_without_base",
        realsense_serials: str = "",
        camera_resolutions: str = "224*224,224*224,224*224",
        robot_control_hz: int = 120,
        robot_observation_hz: int = 30,
        no_camera: bool = False,
        enable_soft_realtime: bool = False,
        bind_cpu_idxs: Optional[list[int]] = None,
        disabled_cameras: Optional[list[str]] = None,
        left_home_joints: Optional[list[float]] = None,
        right_home_joints: Optional[list[float]] = None,
        torso_home_joints: Optional[list[float]] = None,
        gripper_home_positions: Optional[list[float]] = None,
        teleop_config: Optional[dict[str, Any]] = None,
        command_timeout_s: float = 0.3,
        is_dummy: bool = False,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        if structure != "wholebody_without_base":
            raise ValueError(
                "MOZConnection currently supports only 'wholebody_without_base'."
            )
        if robot_control_hz <= 0:
            raise ValueError("robot_control_hz must be positive.")
        if command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive.")
        self._logger = get_logger()
        self._structure = structure
        self._realsense_serials = realsense_serials
        self._camera_resolutions = camera_resolutions
        self._robot_control_hz = int(robot_control_hz)
        self._robot_observation_hz = int(robot_observation_hz)
        self._no_camera = bool(no_camera)
        self._enable_soft_realtime = bool(enable_soft_realtime)
        self._bind_cpu_idxs = bind_cpu_idxs
        self._disabled_cameras = disabled_cameras
        self._left_home_joints = left_home_joints
        self._right_home_joints = right_home_joints
        self._torso_home_joints = torso_home_joints
        self._gripper_home_positions = gripper_home_positions
        self._teleop_config = teleop_config
        self._command_timeout_s = float(command_timeout_s)
        self._is_dummy = bool(is_dummy)
        self._image_size = tuple(image_size)
        self._robot: Any = None
        self._motion_enabled = False
        self._teleop_started = False
        self._stream_stop = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._target_lock = threading.Lock()
        self._latest_target: Optional[dict[str, np.ndarray]] = None
        self._latest_target_at = -float("inf")
        self._hold_target: Optional[dict[str, np.ndarray]] = None
        self._stream_error: Optional[BaseException] = None

    @property
    def observation_features(self) -> Features:
        """Describe the unwrapped SDK snapshot returned by this part."""
        return {
            "right_tcp_pose": {"shape": (6,), "dtype": "float32"},
            "right_joint_positions": {"shape": (7,), "dtype": "float32"},
            "right_gripper": {"shape": (1,), "dtype": "float32"},
        }

    @property
    def action_features(self) -> Features:
        """Describe the complete MOZ command accepted by this part."""
        return {"command": {}}

    def _open(self) -> Any:
        if self._is_dummy:
            self._robot = _DummyMOZRobot(self._image_size)
            self._robot.connect()
            return self._robot

        try:
            from mozrobot import MOZ1Robot, MOZ1RobotConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MOZ SDK is unavailable. Install the MOZ host dependencies and make "
                "RLinf/packages available to the MOZ Python environment."
            ) from exc

        teleop_device = None
        if self._teleop_config is not None:
            from mozrobot.teleoperations.moz1_teleop_configs import MozTeleopConfig

            teleop_device = MozTeleopConfig(**self._teleop_config)

        config = MOZ1RobotConfig(
            realsense_serials=self._realsense_serials,
            camera_resolutions=self._camera_resolutions,
            structure=self._structure,
            robot_control_hz=self._robot_control_hz,
            robot_observation_hz=self._robot_observation_hz,
            no_camera=self._no_camera,
            enable_soft_realtime=self._enable_soft_realtime,
            bind_cpu_idxs=self._bind_cpu_idxs,
            disabled_cameras=self._disabled_cameras,
            teleop_device=teleop_device,
        )
        robot = MOZ1Robot(config)
        # A connection is read-only until the task explicitly resets and opens
        # the motion gate. OpenPI keeps its historic auto-reset default.
        robot.connect(auto_reset=False)
        self._robot = robot
        return robot

    def _release(self, device: Any) -> None:
        del device
        self.disable_motion()
        robot, self._robot = self._robot, None
        if robot is None:
            return
        try:
            robot.stop_teleop()
        except Exception:  # noqa: BLE001 - best effort during hardware cleanup
            self._logger.exception("MOZ teleop did not stop cleanly")
        try:
            robot.disable_external_following_mode()
        except Exception:  # noqa: BLE001 - best effort during hardware cleanup
            self._logger.exception("MOZ external following did not stop cleanly")
        robot.disconnect()

    def get_state(self) -> dict[str, Any]:
        """Return the current MOZ snapshot as a flat dictionary."""
        snapshot = self.get_snapshot()
        return {
            "right_tcp_pose": snapshot.right_tcp_pose,
            "right_joint_positions": snapshot.right_joint_positions,
            "right_gripper": np.asarray([snapshot.right_gripper], dtype=np.float32),
        }

    def get_observation(self) -> Observation:
        """Return the standard part observation contract."""
        return self.get_state()

    def send_action(self, action: Action) -> Observation:
        """Queue one complete, already validated MOZ command."""
        if set(action) != {"command"}:
            raise KeyError("MOZConnection accepts exactly one 'command' action field.")
        command = action["command"]
        if not isinstance(command, Mapping):
            raise TypeError("MOZ command must be a mapping of SDK command fields.")
        self.submit_target(command)
        return {"command": command}

    def get_snapshot(self) -> MOZSnapshot:
        """Read one state/image snapshot without exposing SDK field names."""
        robot = self._require_robot()
        state = robot.capture_robot_observation()
        frames = {} if self._no_camera else robot.capture_images()
        return MOZSnapshot(
            right_tcp_pose=self._vector(state, "rightarm_state_cart_pos", 6),
            right_joint_positions=self._vector(
                state, "rightarm_state_joint_pos", 7
            ),
            right_gripper=self._scalar(state, "rightarm_gripper_state_pos"),
            left_tcp_pose=self._vector(state, "leftarm_state_cart_pos", 6),
            left_gripper=self._scalar(state, "leftarm_gripper_state_pos"),
            torso_tcp_pose=self._optional_vector(state, "torso_state_cart_pos", 6),
            frames={name: np.asarray(frame).copy() for name, frame in frames.items()},
        )

    def reset_to_home(self) -> None:
        """Move to the calibrated joint home before enabling external control."""
        robot = self._require_robot()
        self.disable_motion()
        if not all(
            value is not None
            for value in (
                self._left_home_joints,
                self._right_home_joints,
                self._torso_home_joints,
                self._gripper_home_positions,
            )
        ):
            raise ValueError(
                "MOZ reset requires left/right/torso joint homes and two gripper homes."
            )
        ok = robot.reset_robot_positions(
            left_arm_joints=self._left_home_joints,
            right_arm_joints=self._right_home_joints,
            torso_joints=self._torso_home_joints,
            gripper_positions=self._gripper_home_positions,
        )
        if ok is False:
            raise RuntimeError("MOZ rejected the calibrated reset command.")

    def enable_motion(self) -> None:
        """Enable external following and start the sole local command stream."""
        if self._motion_enabled:
            return
        robot = self._require_robot()
        self._hold_target = self.hold_target(self.get_snapshot())
        robot.enable_external_following_mode()
        self._motion_enabled = True
        self._stream_error = None
        if self._is_dummy:
            return
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="MOZCommandStreamer",
            daemon=True,
        )
        self._stream_thread.start()

    def disable_motion(self) -> None:
        """Stop streaming and leave the MOZ controller out of following mode."""
        self._motion_enabled = False
        self._stream_stop.set()
        thread, self._stream_thread = self._stream_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, 2.0 / self._robot_control_hz))
        robot = self._robot
        if robot is not None:
            try:
                robot.disable_external_following_mode()
            except Exception:  # noqa: BLE001 - cleanup must remain idempotent
                self._logger.exception("MOZ external following did not stop cleanly")

    def submit_target(self, target: Mapping[str, Any]) -> None:
        """Replace the command held by the local stream with a complete target."""
        if not self._motion_enabled:
            raise RuntimeError("MOZ motion is disabled; reset and open the motion gate first.")
        self.raise_if_stream_failed()
        normalized = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in target.items()
        }
        with self._target_lock:
            self._latest_target = normalized
            self._hold_target = normalized
            self._latest_target_at = time.monotonic()
        if self._is_dummy:
            self._send(normalized)

    def hold_target(self, snapshot: MOZSnapshot) -> dict[str, np.ndarray]:
        """Build one full SDK action that holds every enabled non-base part."""
        command = {
            "leftarm_cmd_cart_pos": snapshot.left_tcp_pose.copy(),
            "leftarm_gripper_cmd_pos": np.asarray([snapshot.left_gripper], dtype=np.float32),
            "rightarm_cmd_cart_pos": snapshot.right_tcp_pose.copy(),
            "rightarm_gripper_cmd_pos": np.asarray([snapshot.right_gripper], dtype=np.float32),
        }
        if snapshot.torso_tcp_pose is not None:
            command["torso_cmd_cart_pos"] = snapshot.torso_tcp_pose.copy()
        return command

    def read_native_teleop(self) -> tuple[MOZSnapshot, dict[str, np.ndarray]]:
        """Read MOZ native teleop targets for one policy-format transition."""
        robot = self._require_robot()
        if not self._teleop_started:
            robot.start_teleop(is_dagger_mode=False)
            self._teleop_started = True
        observation, target = robot.teleop_step(record_data=True, record_image=False)
        if observation is None or target is None:
            raise RuntimeError("MOZ native teleop did not return a command target.")
        snapshot = MOZSnapshot(
            right_tcp_pose=self._vector(observation, "rightarm_state_cart_pos", 6),
            right_joint_positions=self._vector(
                observation, "rightarm_state_joint_pos", 7
            ),
            right_gripper=self._scalar(observation, "rightarm_gripper_state_pos"),
            left_tcp_pose=self._vector(observation, "leftarm_state_cart_pos", 6),
            left_gripper=self._scalar(observation, "leftarm_gripper_state_pos"),
            torso_tcp_pose=self._optional_vector(observation, "torso_state_cart_pos", 6),
        )
        return snapshot, {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in target.items()
        }

    def stop_native_teleop(self) -> None:
        """Stop the SDK teleop source without disconnecting the robot."""
        if not self._teleop_started:
            return
        self._require_robot().stop_teleop()
        self._teleop_started = False

    def raise_if_stream_failed(self) -> None:
        """Surface an asynchronous control failure on the caller's step."""
        if self._stream_error is not None:
            raise RuntimeError("MOZ command streamer stopped after an SDK error.") from self._stream_error

    def _stream_loop(self) -> None:
        period = 1.0 / self._robot_control_hz
        deadline = time.monotonic()
        while not self._stream_stop.is_set():
            deadline += period
            try:
                command = self._command_for_stream()
                if command is not None:
                    self._send(command)
            except BaseException as error:  # noqa: BLE001 - surfaced from step()
                self._stream_error = error
                self._logger.exception("MOZ command streamer stopped")
                self._stream_stop.set()
                return
            self._stream_stop.wait(max(0.0, deadline - time.monotonic()))

    def _command_for_stream(self) -> Optional[dict[str, np.ndarray]]:
        with self._target_lock:
            if self._hold_target is None:
                return None
            fresh = time.monotonic() - self._latest_target_at <= self._command_timeout_s
            command = self._latest_target if fresh else self._hold_target
            return {name: value.copy() for name, value in command.items()}

    def _send(self, command: Mapping[str, np.ndarray]) -> None:
        self._require_robot().send_action(command, action_time=time.perf_counter())

    def _require_robot(self) -> Any:
        if not self.is_connected or self._robot is None:
            raise RuntimeError("MOZConnection is not connected. Call connect() first.")
        return self._robot

    @staticmethod
    def _vector(source: Mapping[str, Any], name: str, size: int) -> np.ndarray:
        value = np.asarray(source.get(name, np.zeros(size)), dtype=np.float32).reshape(-1)
        if value.size < size:
            value = np.pad(value, (0, size - value.size))
        return value[:size].copy()

    @staticmethod
    def _optional_vector(
        source: Mapping[str, Any], name: str, size: int
    ) -> Optional[np.ndarray]:
        if name not in source:
            return None
        return MOZConnection._vector(source, name, size)

    @staticmethod
    def _scalar(source: Mapping[str, Any], name: str) -> float:
        value = np.asarray(source.get(name, [0.0]), dtype=np.float32).reshape(-1)
        return float(value[0]) if value.size else 0.0
