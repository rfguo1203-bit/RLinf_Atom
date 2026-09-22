# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""MOZ robot registration and scheduler hardware configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..discovery import RobotConfig
from ..robot import Robot


class MOZRobot(Robot):
    """MOZ robot composed around one atomic full-body SDK connection."""

    ROBOT_TYPE = "MOZ"

    @classmethod
    def build(cls, **kwargs: object) -> "MOZRobot":
        """Build the deferred MOZ connection owned by this robot."""
        from ..parts.arms.moz import MOZConnection

        return cls(moz=MOZConnection(**kwargs))


@dataclass
class MOZRobotConfig(RobotConfig):
    """Node-local MOZ SDK and calibrated reset configuration."""

    structure: str = "wholebody_without_base"
    realsense_serials: str = ""
    camera_resolutions: str = "224*224,224*224,224*224"
    robot_control_hz: int = 120
    robot_observation_hz: int = 30
    no_camera: bool = False
    enable_soft_realtime: bool = False
    bind_cpu_idxs: Optional[list[int]] = None
    disabled_cameras: Optional[list[str]] = None
    left_home_joints: Optional[list[float]] = None
    right_home_joints: Optional[list[float]] = None
    torso_home_joints: Optional[list[float]] = None
    gripper_home_positions: Optional[list[float]] = None
    teleop_config: Optional[dict[str, Any]] = None
    command_timeout_s: float = 0.3
    image_size: list[int] = field(default_factory=lambda: [224, 224])

    def __post_init__(self) -> None:
        """Validate values that are safe to validate without opening hardware."""
        if not isinstance(self.node_rank, int):
            raise TypeError("'node_rank' in MOZ config must be an integer.")
        if self.structure != "wholebody_without_base":
            raise ValueError(
                "MOZ RLPD supports only structure='wholebody_without_base'."
            )
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("MOZ image_size must contain positive [height, width].")
        if self.robot_control_hz <= 0 or self.robot_observation_hz <= 0:
            raise ValueError("MOZ control and observation frequencies must be positive.")
        if self.command_timeout_s <= 0:
            raise ValueError("MOZ command_timeout_s must be positive.")


MOZRobot.register_type(MOZRobotConfig)
