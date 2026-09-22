import logging
import threading
import time
import random
from enum import Enum
from typing import Optional, Dict, Any

from mozrobot.step_control.cl57r_driver import CL57RDriver, HomeMode, DriverError
from mozrobot.step_control.step_control_configs import StepControlConfig

logger = logging.getLogger(__name__)

class StepMode(Enum):
    FIXED_STEP = 0
    RANDOM_STEP = 1

class TriggerMode(Enum):
    FIXED_TRIGGER = 0
    RANDOM_TRIGGER = 1
    NO_TRIGGER = 2

class StepAxisController:
    def __init__(self, config: StepControlConfig):
        """
        Initialize the Step Axis Controller.
        
        Args:
            config: Configuration object containing all step control parameters
        """
        self.config = config
        logging.info(f"Initializing Step Axis Controller with config: {config}")
        
        # Extract parameters from config
        self.serial_port = config.port
        self.baudrate = config.baudrate
        self.slave_id = config.slave_id

        self.driver: Optional[CL57RDriver] = None # Will be generalized later if other drivers are added
        self.driver_lock = threading.RLock()
        self.is_connected = False

        # Step control parameters from config
        self.step_mode = StepMode[config.step_mode]
        self.fixed_step_value = config.fixed_step_value
        self.random_step_min = int(config.random_step_min*config.step_per_mm)
        self.random_step_max = int(config.random_step_max*config.step_per_mm)
        self.trigger_mode = TriggerMode[config.trigger_mode]
        self.fixed_trigger_value = config.fixed_trigger_value
        self.random_trigger_min = config.random_trigger_min
        self.random_trigger_max = config.random_trigger_max
        self.default_speed = config.default_speed
        self.default_accel_time = config.default_accel_time
        self.default_decel_time = config.default_decel_time
        self.default_home_speed = config.default_home_speed
        self.default_home_creep_speed = config.default_home_creep_speed

        # Internal state variables
        self._gather_data_count_temp = -1 # Used for deduplication
        self._trigger_value: int = 0
        self._step_value: int = 0
        self._init_values: bool = False # Flag if step and trigger values have been initialized
        self._random = random.Random(config.seed) if config.seed is not None else random.Random()

        logger.info(f'Step Axis Controller initialized: Port={self.serial_port}, Baudrate={self.baudrate}, Slave ID={self.slave_id}')

    def _make_driver(self, driver_type: str, port: str, baudrate: int, slave_id: int):
        """
        Factory method to create a specific step motor driver instance.
        """
        if driver_type == "cl57r":
            return CL57RDriver(port=port, baudrate=baudrate, slave_id=slave_id, timeout=1.0)
        else:
            raise ValueError(f"Unsupported step motor driver type: {driver_type}")

    def connect(self) -> bool:
        """Attempts to connect the driver and enable the motor."""
        if self.is_connected:
            logger.info("Driver already connected.")
            return True

        try:
            with self.driver_lock:
                if self.driver:
                    self.driver.close()
                
                self.driver = self._make_driver(self.config.type, self.serial_port, self.baudrate, self.slave_id)

                # Test connection
                status = self.driver.get_status()
                if status is not None:
                    self.is_connected = True
                    logger.info('Driver connected successfully.')

                    # Enable motor
                    if self.motor_enable():
                        logger.info('Motor enabled.')
                        return True
                    else:
                        logger.error('Failed to enable motor.')
                        self.disconnect()
                        return False
                else:
                    raise DriverError("Failed to get driver status, connection failed.")

        except Exception as e:
            self.is_connected = False
            if self.driver:
                try:
                    self.driver.close()
                except Exception as close_e:
                    logger.error(f"Error closing driver: {close_e}")
                finally:
                    self.driver = None
            logger.error(f'Driver connection failed: {e}')
            return False

    def disconnect(self) -> None:
        """Disconnects the driver and releases the motor."""
        logger.info('Closing driver connection...')
        with self.driver_lock:
            if self.driver:
                try:
                    self.motor_disable()  # Attempt to stop motion and release motor
                    self.driver.close()
                    logger.info('Driver connection closed.')
                except Exception as e:
                    logger.error(f'Error closing driver: {e}')
                finally:
                    self.driver = None
                    self.is_connected = False
    
    def motor_enable(self) -> bool:
        """Enables the motor."""
        if not self.is_connected or not self.driver:
            logger.warning("Driver not connected, cannot enable motor.")
            return False
        with self.driver_lock:
            return self.driver.motor_enable()

    def motor_disable(self) -> bool:
        """Disables the motor (releases)."""
        if not self.is_connected or not self.driver:
            logger.warning("Driver not connected, cannot disable motor.")
            return False
        with self.driver_lock:
            return self.driver.motor_disable()

    def clear_alarm(self) -> bool:
        """Clears current alarms."""
        if not self.is_connected or not self.driver:
            logger.warning("Driver not connected, cannot clear alarms.")
            return False
        with self.driver_lock:
            return self.driver.clear_alarm()

    def home_motor(self, home_mode: HomeMode = HomeMode.NEGATIVE_HOME ) -> bool:
        """
        Executes homing operation.
        :param home_mode: Homing mode (HomeMode enum)
        :return: True if homing started successfully, False otherwise
        """
        logger.info(f'Received homing request - Mode: {home_mode.name}, Speed: {self.default_home_speed}, Creep Speed: {self.default_home_creep_speed}')
        if not self.is_connected or not self.driver:
            logger.error("Driver not connected, cannot perform homing.")
            return False

        try:
            with self.driver_lock:
                if self.default_home_speed > 0:
                    self.driver.set_home_speed(self.default_home_speed)
                if self.default_home_creep_speed > 0:
                    self.driver.set_home_creep_speed(self.default_home_creep_speed)

                success = self.driver.start_homing(home_mode)
                if success:
                    logger.info(f'Homing started successfully: {home_mode.name}')
                else:
                    logger.error("Failed to start homing.")
                return success
        except Exception as e:
            logger.error(f'Homing operation failed: {e}')
            return False

    def move_relative(self, target_position: int, speed: int = 500, accel_time: int = 100, decel_time: int = 100) -> bool:
        """
        Executes relative position movement.
        :param target_position: Target position (pulses, -2147483648 ~ 2147483647)
        :param speed: Speed (r/min, 0-3000)
        :param accel_time: Acceleration time (ms, 0-2000)
        :param decel_time: Deceleration time (ms, 0-2000)
        :return: True if relative movement started successfully, False otherwise
        """
        logger.info(f'Received relative move request - Target: {target_position}, Speed: {speed}')
        if not self.is_connected or not self.driver:
            logger.error("Driver not connected, cannot perform relative move.")
            return False

        try:
            with self.driver_lock:
                self.driver.set_position_speed(speed)
                self.driver.set_position_acceleration_time(accel_time)
                self.driver.set_position_deceleration_time(decel_time)
                
                success1 = self.driver.set_position_target(target_position)
                success2 = self.driver.start_relative_positioning()
                
                if success1 and success2:
                    logger.info(f'Relative move started successfully: Target {target_position} pulses')
                else:
                    logger.error("Failed to start relative move.")
                return success1 and success2
        except Exception as e:
            logger.error(f'Relative move operation failed: {e}')
            return False

    def move_absolute(self, target_position: int, speed: int = 200, accel_time: int = 100, decel_time: int = 100) -> bool:
        """
        Executes absolute position movement.
        :param target_position: Target position (pulses, -2147483648 ~ 2147483647)
        :param speed: Speed (r/min, 0-3000)
        :param accel_time: Acceleration time (ms, 0-2000)
        :param decel_time: Deceleration time (ms, 0-2000)
        :return: True if absolute movement started successfully, False otherwise
        """
        logger.info(f'Received absolute move request - Target: {target_position}, Speed: {speed}')
        if not self.is_connected or not self.driver:
            logger.error("Driver not connected, cannot perform absolute move.")
            return False

        try:
            with self.driver_lock:
                self.driver.set_position_speed(speed)
                self.driver.set_position_acceleration_time(accel_time)
                self.driver.set_position_deceleration_time(decel_time)
                
                success1 = self.driver.set_position_target(target_position)
                success2 = self.driver.start_absolute_positioning()
                
                if success1 and success2:
                    logger.info(f'Absolute move started successfully: Target {target_position} pulses')
                else:
                    logger.error("Failed to start absolute move.")
                return success1 and success2
        except Exception as e:
            logger.error(f'Absolute move operation failed: {e}')
            return False

    def get_current_position(self) -> Optional[int]:
        """Gets the current position."""
        if not self.is_connected or not self.driver:
            logger.warning("Driver not connected, cannot get position.")
            return None
        with self.driver_lock:
            return self.driver.get_current_position()

    def get_status_info(self) -> Dict[str, Any]:
        """Gets and returns the driver's current status and alarm information."""
        status_info = {
            'is_connected': self.is_connected,
            'is_homed': False,
            'is_moving': False,
            'in_position': False,
            'has_fault': False,
            'current_position': None,
            'current_speed': None,
            'alarm_code': None,
            'alarm_name': None,
            'alarm_description': None,
            'message': 'Driver not connected or no data'
        }
        if not self.is_connected or not self.driver:
            return status_info

        try:
            with self.driver_lock:
                status = self.driver.get_status()
                position = self.driver.get_current_position()
                speed = self.driver.get_current_speed()
                alarm = self.driver.get_alarm_status()

                if status:
                    status_info['is_homed'] = status.get('homing_done', False)
                    status_info['is_moving'] = status.get('motor_running', False)
                    status_info['in_position'] = status.get('in_position', False)
                    status_info['has_fault'] = status.get('fault', False)
                    status_info['current_position'] = position
                    status_info['current_speed'] = speed
                    
                    if alarm and alarm['has_alarm']:
                        status_info['has_fault'] = True
                        status_info['alarm_code'] = alarm.get('alarm_code')
                        status_info['alarm_name'] = alarm.get('alarm_name')
                        status_info['alarm_description'] = alarm.get('alarm_description')
                        status_info['message'] = f"Fault: {status_info['alarm_description']}"
                    elif status_info['is_moving']:
                        status_info['message'] = "Moving"
                    elif status_info['in_position']:
                        status_info['message'] = "In position"
                    else:
                        status_info['message'] = "Ready"
                else:
                    status_info['message'] = "Failed to get driver status."

        except Exception as e:
            logger.error(f'Failed to get status: {e}')
            status_info['message'] = f"Error getting status: {e}"
            status_info['has_fault'] = True
        
        return status_info

    def check_motion_complete(self) -> Dict[str, Any]:
        """Checks if motion is complete."""
        motion_status = {
            'success': False,
            'motion_complete': False,
            'in_position': False,
            'message': 'Driver not connected'
        }
        if not self.is_connected or not self.driver:
            return motion_status
        
        try:
            with self.driver_lock:
                status = self.driver.get_status()
                if status:
                    motion_status['success'] = True
                    motion_status['in_position'] = status.get('in_position', False)
                    motion_status['motion_complete'] = not status.get('motor_running', False)
                    
                    if motion_status['in_position']:
                        motion_status['message'] = "In position"
                    elif motion_status['motion_complete']:
                        motion_status['message'] = "Motion complete"
                    else:
                        motion_status['message'] = "Moving"
                else:
                    motion_status['message'] = "Failed to get status"
        except Exception as e:
            logger.error(f'Failed to check motion status: {e}')
            motion_status['message'] = f"Error checking motion status: {e}"
        return motion_status

    def check_homed(self) -> Dict[str, Any]:
        """Checks if homing is complete."""
        homed_status = {
            'success': False,
            'is_homed': False,
            'message': 'Driver not connected'
        }
        if not self.is_connected or not self.driver:
            return homed_status
        
        try:
            with self.driver_lock:
                status = self.driver.get_status()
                if status:
                    homed_status['success'] = True
                    homed_status['is_homed'] = status.get('homing_done', False)
                    homed_status['message'] = "Homing complete" if homed_status['is_homed'] else "Not homed"
                else:
                    homed_status['message'] = "Failed to get status"
        except Exception as e:
            logger.error(f'Failed to check homing status: {e}')
            homed_status['message'] = f"Error checking homing status: {e}"
        return homed_status

    def _generate_random_in_range(self, min_val: int, max_val: int) -> int:
        """Generates a random integer within the specified range (inclusive of min_val and max_val)."""
        if min_val > max_val:
            min_val, max_val = max_val, min_val # Ensure min <= max
        return self._random.randint(min_val, max_val)

    def execute_periodic_adjustment(self, gather_data_count: int) -> None:
        """
        Periodically triggers the step axis adjustment based on the data collection counter.
        This function should be called after each data collection frame.
        :param gather_data_count: Current data collection count.
        """
        with self.driver_lock:
            # Prevent duplicate execution
            if self._gather_data_count_temp == gather_data_count:
                return
            self._gather_data_count_temp = gather_data_count

            if not self._init_values:  # Initialize step control parameters
                # Initialize trigger value
                if self.trigger_mode == TriggerMode.FIXED_TRIGGER:
                    self._trigger_value = self.fixed_trigger_value
                elif self.trigger_mode == TriggerMode.RANDOM_TRIGGER:
                    self._trigger_value = self._generate_random_in_range(
                        self.random_trigger_min,
                        self.random_trigger_max
                    )
                elif self.trigger_mode == TriggerMode.NO_TRIGGER:
                    self._trigger_value = 0

                # Initialize step value
                if self.step_mode == StepMode.FIXED_STEP:
                    self._step_value = self.fixed_step_value
                elif self.step_mode == StepMode.RANDOM_STEP:
                    self._step_value = self._generate_random_in_range(
                        self.random_step_min,
                        self.random_step_max
                    )
                
                # Ensure trigger value and step value are non-zero. If zero, warn and skip initialization.
                if self._trigger_value <= 0 or self._step_value == 0:
                    logger.warning(f"Invalid trigger value ({self._trigger_value}) or step value ({self._step_value}), skipping periodic adjustment initialization.")
                    self._init_values = False # Ensure next attempt will still try to initialize
                    return

                self._init_values = True
                logger.info(f"Step axis periodic adjustment parameters initialized: Trigger Value={self._trigger_value}, Step Value={self._step_value}")

            if self._trigger_value > 0 and gather_data_count % self._trigger_value == 0:
                # Trigger step axis movement
                logger.info(f"Triggering step axis movement: Trigger Value={self._trigger_value}, Step Value={self._step_value}, Data Count={gather_data_count}")
                success = self.move_absolute(
                    target_position=self._step_value,
                    speed=self.default_speed,
                    accel_time=self.default_accel_time,
                    decel_time=self.default_decel_time
                )
                if success:
                    self._init_values = False  # Reset after successful movement to re-initialize parameters next time
                else:
                    logger.error("Failed to trigger step axis movement.")
            else:
                logger.debug(f"Not at step axis trigger point: Data Count={gather_data_count}, Trigger Value={self._trigger_value}")

    def __enter__(self):
        """Supports the 'with' statement."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Supports the 'with' statement."""
        self.disconnect() 