from mozrobot.step_control.step_control_configs import StepControlConfig, CL57RStepControlConfig
from mozrobot.step_control.step_axis_controller import StepAxisController

def make_step_axis_controller_from_config(config: StepControlConfig):
    if isinstance(config, CL57RStepControlConfig):
        return StepAxisController(config)
    else:
        raise ValueError(f"Unknown step control device type: {config.type}") 