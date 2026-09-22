"""MOZRobot teleoperation modules"""

from .moz1_teleop_configs import TeleopDeviceConfig

# Try to import teleop adapter, but don't fail if ROS2 is not available
try:
    from .moz_teleop_adapter import *
except ImportError as e:
    import logging
    logging.warning(f"Could not import moz_teleop_adapter: {e}")
    # Define minimal stubs for missing classes
    class MozTeleopAdapter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ROS2 not available. Cannot use MozTeleopAdapter.")

__all__ = [
    "TeleopDeviceConfig",
]