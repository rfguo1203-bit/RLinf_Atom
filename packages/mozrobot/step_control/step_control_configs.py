from dataclasses import dataclass
from typing import Literal, Optional
import draccus

@dataclass
class StepControlConfig(draccus.ChoiceRegistry):
    """Base configuration class for step control devices."""
    
    # Common parameters for all step control devices
    port: str = "/dev/cl57r"
    baudrate: int = 115200
    slave_id: int = 1
    
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

@StepControlConfig.register_subclass("cl57r")
@dataclass 
class CL57RStepControlConfig(StepControlConfig):
    """Configuration for CL57R step motor driver."""
    step_per_mm: float = 247.581
    max_mm_value: int = 200
    # Step control parameters
    step_mode: Literal["FIXED_STEP", "RANDOM_STEP"] = "RANDOM_STEP"
    fixed_step_value: int = 100
    random_step_min: int = 50
    random_step_max: int = 150
    
    # Trigger control parameters  
    trigger_mode: Literal["FIXED_TRIGGER", "RANDOM_TRIGGER", "NO_TRIGGER"] = "RANDOM_TRIGGER"
    fixed_trigger_value: int = 10
    random_trigger_min: int = 5
    random_trigger_max: int = 15
    
    # Motor control parameters
    default_speed: int = 200
    default_accel_time: int = 100
    default_decel_time: int = 100

    # Homing control parameters
    default_home_speed: int = 200
    default_home_creep_speed: int = 100

    # Initial position control parameters（mm）
    initial_position: Optional[float] = None

    # Random seed for reproducibility
    seed: Optional[int] = None 