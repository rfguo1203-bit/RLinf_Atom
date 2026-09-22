from dataclasses import dataclass
import draccus
import abc

@dataclass
class TeleopDeviceConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base config for teleop devices."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

# 基于moz_teleop API的配置类 - 统一自适应配置
@TeleopDeviceConfig.register_subclass("moz_teleop")
@dataclass
class MozTeleopConfig(TeleopDeviceConfig):
    """基于moz_teleop API的统一自适应遥操作配置

    根据teleop_state topic自动适应VR或外骨骼设备，无需区分设备类型
    """
    # ROS通信配置
    ros_service_name: str = "teleop_cmd_service"
    teleop_state_topic: str = "teleop_state"
    mix_command_topic: str = "mx_mix_command"
    gripper_command_topic: str = "mx_gripper_command"
    base_command_topic: str = "mx_base_vel_command"

    # 超时设置
    ros_service_timeout: float = 5.0
    topic_timeout: float = 1.0

    enable_moz_teleop: bool = True