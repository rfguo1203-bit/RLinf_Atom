# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field
from typing import Sequence, Literal, Dict, Optional, Union # Keep Literal for step_mode and trigger_mode

import draccus
import numpy as np

try:
    # Import directly from the file to avoid __init__.py issues
    import importlib.util
    import os
    _teleop_configs_path = os.path.join(os.path.dirname(__file__), 'teleoperations', 'moz1_teleop_configs.py')
    _spec = importlib.util.spec_from_file_location("moz1_teleop_configs", _teleop_configs_path)
    _teleop_configs_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_teleop_configs_module)
    TeleopDeviceConfig = _teleop_configs_module.TeleopDeviceConfig
except ImportError:
    # Define a minimal stub when teleoperations can't be imported
    from dataclasses import dataclass
    @dataclass
    class TeleopDeviceConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ROS2 not available. Cannot use TeleopDeviceConfig.")

from mozrobot.step_control.step_control_configs import StepControlConfig 

@dataclass
class CameraConfig:
    fps: int
    width: int
    height: int


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyRobotConfig(RobotConfig):
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {})
    teleop_device: TeleopDeviceConfig | None = None
    step_control_device: StepControlConfig | None = None

    def __post_init__(self):
        self.cameras = {}
        for camera_name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            self.cameras[camera_name] = CameraConfig(
                fps=30,
                width=224,
                height=224,
            )

@RobotConfig.register_subclass("moz1")
@dataclass
class MOZ1RobotConfig(RobotConfig):
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {})
    # Camera configuration - supports both USB ports and direct serial numbers (auto-detected)
    realsense_serials: Optional[str] = "0000000000,0000000000,0000000000"
    camera_resolutions: str = "320*240,320*240,320*240"
    simu_mode: bool = False
    virtual_robot: bool = False
    no_camera: bool = False  # Disable camera functionality
    # choices: dualarm, wholebody_without_base, wholebody
    structure: str = "dualarm"
    robot_control_hz: int = 120
    robot_observation_hz: int = 30
    teleop_device: Optional[TeleopDeviceConfig] = None
    step_control_device: Optional[StepControlConfig] = None
    enable_driver_logging: bool = False
    # Soft real-time configuration - requires running scripts/setup_rtprio.sh first
    enable_soft_realtime: bool = False
    # CPU affinity configuration - None means no CPU binding
    bind_cpu_idxs: Optional[Sequence[int]] = None
    # Disabled cameras list - allows disabling specific cameras by name
    disabled_cameras: Optional[Sequence[str]] = None

    def __post_init__(self):
        # Validate camera configuration when not using no_camera or simu_mode
        if not self.no_camera and not self.simu_mode and self.realsense_serials is None:
            raise ValueError("realsense_serials must be specified when no_camera=False and simu_mode=False")

        # Validate disabled_cameras parameter
        if self.disabled_cameras is not None:
            valid_camera_names = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
            if isinstance(self.disabled_cameras, str):
                self.disabled_cameras = [self.disabled_cameras]

            invalid_cameras = set(self.disabled_cameras) - valid_camera_names
            if invalid_cameras:
                raise ValueError(f"Invalid camera names in disabled_cameras: {invalid_cameras}. Valid names: {valid_camera_names}")

            # Warn if all cameras are disabled but camera system is required
            if not self.no_camera and not self.simu_mode and set(self.disabled_cameras) == valid_camera_names:
                raise ValueError("Cannot disable all cameras when camera system is required (no_camera=False and simu_mode=False)")

        self.cameras = {}
        for camera_name, camera_resolution in zip(["cam_high", "cam_left_wrist", "cam_right_wrist"], self.camera_resolutions.split(",")):
            self.cameras[camera_name] = CameraConfig(
                fps=self.robot_observation_hz,
                width=int(camera_resolution.split("*")[0]),
                height=int(camera_resolution.split("*")[1]),
            )