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

"""MOZ's existing ROS teleoperation source as an RLinf operator device."""

from __future__ import annotations

from typing import Any, Mapping

from rlinf.robotics.actions import ActionKind
from rlinf.robotics.parts.arms.moz import MOZConnection, MOZSnapshot

from .base import Features, Observation, TeleopAction, TeleopDevice


@TeleopDevice.register("moz_native")
class MOZNativeTeleop(TeleopDevice):
    """Turn the MOZ SDK's native leader target into one environment action.

    The native ROS adapter only provides its current target. The environment
    owns the calibrated action mapping and the MOZ connection remains the
    single command writer, so this device never sends a vendor command itself.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }
    NEEDS = ("moz_connection", "moz_teleop_mapper")
    HOLD_WINDOW = 0.0

    def __init__(self) -> None:
        self._connection: MOZConnection | None = None

    def _open(self) -> object:
        """Open no separate device; MOZ owns the native ROS adapter."""
        return object()

    @property
    def observation_features(self) -> Features:
        """Describe the one native target consumed per policy step."""
        return {"native_target": {}}

    def get_observation(self) -> Observation:
        """Return a placeholder because :meth:`drive` reads with context."""
        return {"native_target": {}}

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Reject the context-free mapping path used by ordinary devices."""
        del reading, context
        raise RuntimeError("MOZNativeTeleop maps targets through drive().")

    def drive(
        self,
        context: Mapping[str, Any],
        reading: Mapping[str, Any] | None = None,
    ) -> TeleopAction:
        """Read one SDK target and express it in the environment action space."""
        del reading
        connection = context["moz_connection"]
        if not isinstance(connection, MOZConnection):
            raise TypeError("MOZ native teleop requires an MOZConnection context.")
        mapper = context["moz_teleop_mapper"]
        if not callable(mapper):
            raise TypeError("MOZ native teleop requires a callable target mapper.")

        self._connection = connection
        snapshot, target = connection.read_native_teleop()
        parts = mapper(snapshot, target)
        return TeleopAction(
            parts=parts,
            driving=True,
            info={"moz_native_teleop": True},
        )

    def _release(self, device: object) -> None:
        """Stop the ROS leader source while retaining normal robot cleanup."""
        del device
        if self._connection is not None:
            self._connection.stop_native_teleop()
            self._connection = None
