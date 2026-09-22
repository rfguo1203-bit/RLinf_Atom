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

"""Right-arm fixed-tabletop pick-and-lift task for the MOZ robot."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics import MOZRobot, MOZRobotConfig, Robot, RobotInfo
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms import MOZConnection, MOZSnapshot
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger


_ACTION_DIM = 7
_STATE_DIM = 14
_FRAME_KEYS = ("head_rgb", "right_wrist_rgb")
_SDK_FRAME_KEYS = {"head_rgb": "cam_high", "right_wrist_rgb": "cam_right_wrist"}


@dataclass
class MozPickLiftConfig:
    """Task parameters and mandatory motion-safety limits for MOZ.

    ``motion_enabled`` stays false until a site-specific configuration supplies
    a verified home pose, Cartesian workspace, and gripper range. The task can
    still connect read-only for camera and schema validation in that state.
    """

    is_dummy: bool = False
    motion_enabled: bool = False
    safety_calibrated: bool = False

    step_frequency: float = 10.0
    max_num_steps: int = 150
    enable_camera_player: bool = False

    translation_action_scale: float = 0.01
    rotation_action_scale: float = 0.05
    workspace_min: Optional[list[float]] = None
    workspace_max: Optional[list[float]] = None
    gripper_position_min: Optional[float] = None
    gripper_position_max: Optional[float] = None
    reset_settle_s: float = 1.0

    def validate_motion(self, hardware: MOZRobotConfig) -> None:
        """Reject any physical run whose calibrated command contract is absent."""
        if self.is_dummy or not self.motion_enabled:
            return
        if not self.safety_calibrated:
            raise ValueError(
                "Set safety_calibrated=true only after reviewing the MOZ home, "
                "workspace, and gripper values on the physical robot."
            )
        if hardware.no_camera:
            raise ValueError("MOZ pick-and-lift requires head and right-wrist RGB.")
        disabled = set(hardware.disabled_cameras or [])
        needed = {"cam_high", "cam_right_wrist"}
        if missing := needed & disabled:
            raise ValueError(f"MOZ task cameras are disabled: {sorted(missing)}.")
        serials = [serial.strip() for serial in hardware.realsense_serials.split(",")]
        if len(serials) != 3 or any(not serial for serial in serials):
            raise ValueError(
                "MOZ realsense_serials must list non-empty head, left wrist, and "
                "right wrist serials."
            )
        homes = {
            "left_home_joints": (hardware.left_home_joints, 7),
            "right_home_joints": (hardware.right_home_joints, 7),
            "torso_home_joints": (hardware.torso_home_joints, 6),
            "gripper_home_positions": (hardware.gripper_home_positions, 2),
        }
        for name, (value, size) in homes.items():
            if value is None or len(value) != size or not np.isfinite(value).all():
                raise ValueError(f"MOZ physical reset requires {name} with {size} values.")
        minimum, maximum = self.workspace_limits()
        if not np.all(minimum < maximum):
            raise ValueError("Each MOZ workspace_min value must be less than workspace_max.")
        gripper_min, gripper_max = self.gripper_limits()
        if not gripper_min < gripper_max:
            raise ValueError("gripper_position_min must be less than gripper_position_max.")
        if self.step_frequency <= 0 or self.max_num_steps <= 0:
            raise ValueError("MOZ step_frequency and max_num_steps must be positive.")
        if self.translation_action_scale <= 0 or self.rotation_action_scale <= 0:
            raise ValueError("MOZ Cartesian action scales must be positive.")

    def workspace_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the configured right-arm Cartesian safety box."""
        minimum = self._vector(self.workspace_min, "workspace_min")
        maximum = self._vector(self.workspace_max, "workspace_max")
        return minimum, maximum

    def gripper_limits(self) -> tuple[float, float]:
        """Return the configured physical range of the right gripper."""
        if self.gripper_position_min is None or self.gripper_position_max is None:
            if self.is_dummy:
                return 0.0, 1.0
            raise ValueError(
                "MOZ physical motion requires gripper_position_min and "
                "gripper_position_max."
            )
        return float(self.gripper_position_min), float(self.gripper_position_max)

    def _vector(self, value: Optional[list[float]], name: str) -> np.ndarray:
        if value is None:
            if self.is_dummy:
                return np.full(6, -1.0 if name.endswith("min") else 1.0)
            raise ValueError(f"MOZ physical motion requires {name} with 6 values.")
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size != 6 or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain six finite Cartesian values.")
        return vector


class MozPickLiftEnv(gym.Env):
    """Drive MOZ's right arm for a fixed tabletop pick-and-lift rollout."""

    TELEOP = ("moz_native",)
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()
    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Optional[RobotInfo[Any]] = None,
        env_idx: int = 0,
    ) -> None:
        self._logger = get_logger()
        self.config = MozPickLiftConfig(**override_cfg)
        self.hardware = get_hardware_config(
            MOZRobotConfig, robot_info, is_dummy=self.config.is_dummy
        )
        self.config.validate_motion(self.hardware)
        self.env_idx = env_idx
        self.node_rank = 0 if worker_info is None else worker_info.cluster_node_rank
        self._num_steps = 0
        self._last_snapshot: MOZSnapshot | None = None
        # The scheduler already places this environment worker on the MOZ
        # node. Keeping the SDK in that worker avoids a second local Ray actor
        # and leaves exactly one process able to command the controller.
        self.robot: Robot = MOZRobot.build(**self._connection_kwargs())
        self.robot.connect()
        self._connection = self.robot.child("moz", MOZConnection)

        self._init_spaces()
        self._last_snapshot = self._connection.get_snapshot()

    @property
    def task_description(self) -> str:
        """Describe the single fixed-object task shown to the policy."""
        return "Pick up the tabletop object with the right arm and lift it."

    def _connection_kwargs(self) -> dict[str, Any]:
        return {
            "structure": self.hardware.structure,
            "realsense_serials": self.hardware.realsense_serials,
            "camera_resolutions": self.hardware.camera_resolutions,
            "robot_control_hz": self.hardware.robot_control_hz,
            "robot_observation_hz": self.hardware.robot_observation_hz,
            "no_camera": self.hardware.no_camera,
            "enable_soft_realtime": self.hardware.enable_soft_realtime,
            "bind_cpu_idxs": self.hardware.bind_cpu_idxs,
            "disabled_cameras": self.hardware.disabled_cameras,
            "left_home_joints": self.hardware.left_home_joints,
            "right_home_joints": self.hardware.right_home_joints,
            "torso_home_joints": self.hardware.torso_home_joints,
            "gripper_home_positions": self.hardware.gripper_home_positions,
            "teleop_config": self.hardware.teleop_config,
            "command_timeout_s": self.hardware.command_timeout_s,
            "is_dummy": self.config.is_dummy,
            "image_size": tuple(self.hardware.image_size),
        }

    def _init_spaces(self) -> None:
        height, width = self._image_shape()
        self.action_space = gym.spaces.Box(
            low=-np.ones(_ACTION_DIM, dtype=np.float32),
            high=np.ones(_ACTION_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                        "right_joint_positions": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                        "right_tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(height, width, 3), dtype=np.uint8
                        )
                        for name in _FRAME_KEYS
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _image_shape(self) -> tuple[int, int]:
        size = np.asarray(self.hardware.image_size, dtype=int).reshape(-1)
        if size.size != 2 or np.any(size <= 0):
            raise ValueError("MOZ image_size must contain positive [height, width].")
        return int(size[0]), int(size[1])

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Describe the right Cartesian delta and normalized gripper command."""
        return (
            ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
            ActionPart("end_effector", 1, ActionKind.GRIPPER),
        )

    def get_moz_connection(self) -> MOZConnection:
        """Expose the local SDK owner to the native teleop adapter."""
        return self._connection

    def get_moz_teleop_mapper(self) -> Any:
        """Expose the calibrated native-target mapping to the teleop adapter."""
        return self._native_target_to_action

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset into a safe home only when physical motion is enabled."""
        del options
        seed_sampled_spaces(seed, self._base_observation_space)
        self._num_steps = 0
        if self.config.is_dummy:
            self._connection.enable_motion()
        elif self.config.motion_enabled:
            self._connection.stop_native_teleop()
            self._connection.reset_to_home()
            time.sleep(self.config.reset_settle_s)
            self._connection.enable_motion()
        self._last_snapshot = self._connection.get_snapshot()
        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one bounded policy action through the local MOZ command stream."""
        if not self.config.is_dummy and not self.config.motion_enabled:
            raise RuntimeError(
                "MOZ motion is locked. Set calibrated limits and motion_enabled=true "
                "only for an approved physical run."
            )
        if self._last_snapshot is None:
            self._last_snapshot = self._connection.get_snapshot()
        start = time.monotonic()
        normalized = np.asarray(action, dtype=np.float64).reshape(-1)
        if normalized.size != _ACTION_DIM:
            raise ValueError(f"MOZ action must have {_ACTION_DIM} values, got {normalized.size}.")
        normalized = np.clip(normalized, self.action_space.low, self.action_space.high)

        target = self._target_from_action(self._last_snapshot, normalized)
        self._connection.submit_target(target)
        self._num_steps += 1
        if not self.config.is_dummy:
            time.sleep(max(0.0, 1.0 / self.config.step_frequency - (time.monotonic() - start)))
        self._last_snapshot = self._connection.get_snapshot()
        truncated = self._num_steps >= self.config.max_num_steps
        return self._get_observation(), 0.0, False, truncated, {}

    def _target_from_action(
        self, snapshot: MOZSnapshot, action: np.ndarray
    ) -> dict[str, np.ndarray]:
        target = self._connection.hold_target(snapshot)
        target["rightarm_cmd_cart_pos"] = self._pose_from_delta(snapshot, action[:6])
        target["rightarm_gripper_cmd_pos"] = np.asarray(
            [self._gripper_from_action(float(action[6]))], dtype=np.float32
        )
        return target

    def _pose_from_delta(self, snapshot: MOZSnapshot, action: np.ndarray) -> np.ndarray:
        current = np.asarray(snapshot.right_tcp_pose, dtype=np.float64)
        delta = np.asarray(action, dtype=np.float64)
        position = current[:3] + delta[:3] * self.config.translation_action_scale
        rotation = (
            Rotation.from_rotvec(delta[3:] * self.config.rotation_action_scale)
            * Rotation.from_rotvec(current[3:])
        ).as_rotvec()
        minimum, maximum = self.config.workspace_limits()
        return np.clip(np.concatenate((position, rotation)), minimum, maximum).astype(
            np.float32
        )

    def _gripper_from_action(self, action: float) -> float:
        minimum, maximum = self.config.gripper_limits()
        return float(minimum + (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (maximum - minimum))

    def _native_target_to_action(
        self, snapshot: MOZSnapshot, target: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        pose = np.asarray(
            target.get("rightarm_cmd_cart_pos", snapshot.right_tcp_pose), dtype=np.float64
        ).reshape(6)
        current = np.asarray(snapshot.right_tcp_pose, dtype=np.float64)
        translation = (pose[:3] - current[:3]) / self.config.translation_action_scale
        rotation = (
            Rotation.from_rotvec(pose[3:]) * Rotation.from_rotvec(current[3:]).inv()
        ).as_rotvec() / self.config.rotation_action_scale
        target_gripper = float(
            np.asarray(
                target.get("rightarm_gripper_cmd_pos", [snapshot.right_gripper]),
                dtype=np.float64,
            ).reshape(-1)[0]
        )
        gripper_min, gripper_max = self.config.gripper_limits()
        gripper = 2.0 * (target_gripper - gripper_min) / (gripper_max - gripper_min) - 1.0
        return {
            "arm": np.clip(np.concatenate((translation, rotation)), -1.0, 1.0).astype(
                np.float32
            ),
            "end_effector": np.asarray([np.clip(gripper, -1.0, 1.0)], dtype=np.float32),
        }

    def _get_observation(self) -> dict[str, Any]:
        if self._last_snapshot is None:
            raise RuntimeError("MOZ observation requested before the first snapshot.")
        snapshot = self._last_snapshot
        height, width = self._image_shape()
        frames: dict[str, np.ndarray] = {}
        for name, sdk_name in _SDK_FRAME_KEYS.items():
            if sdk_name not in snapshot.frames:
                raise RuntimeError(
                    f"MOZ camera {sdk_name!r} did not produce the required {name!r} frame."
                )
            frame = np.asarray(snapshot.frames[sdk_name], dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"MOZ frame {sdk_name!r} has shape {frame.shape}; expected "
                    f"{(height, width, 3)}. Align camera_resolutions and image_size."
                )
            frames[name] = frame.copy()
        return {
            "state": {
                "right_gripper": np.asarray([snapshot.right_gripper], dtype=np.float32),
                "right_joint_positions": snapshot.right_joint_positions.astype(
                    np.float32, copy=True
                ),
                "right_tcp_pose": snapshot.right_tcp_pose.astype(np.float32, copy=True),
            },
            "frames": frames,
        }

    def close(self) -> None:
        """Stop following mode and release the node-local MOZ SDK session."""
        self.robot.disconnect()
        super().close()
