import serial
import struct
import time
import logging
from enum import Enum
from typing import Optional, List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriverError(Exception):
    """驱动器异常类"""
    pass

class AlarmCode(Enum):
    """报警代码枚举"""
    OVERCURRENT = 0x01      # 过流
    OVERVOLTAGE = 0x02      # 过压  
    UNDERVOLTAGE = 0x03     # 欠压
    MEMORY_ERROR = 0x04     # 存储器读写错误
    POSITION_ERROR = 0x05   # 位置超差报警

class HomeMode(Enum):
    """回零模式枚举"""
    NEGATIVE_LIMIT = 17     # 负限位模式
    POSITIVE_LIMIT = 18     # 正限位模式
    POSITIVE_HOME = 24      # 正向原点模式
    NEGATIVE_HOME = 29      # 负向原点模式
    CURRENT_POSITION = 35   # 设置当前位置为原点

class CL57RDriver:
    def __init__(self, port: str, baudrate: int = 115200, slave_id: int = 1, timeout: float = 1.0):
        """
        初始化步进驱动器
        :param port: 串口号 (如 'COM1' 或 '/dev/ttyUSB0')
        :param baudrate: 波特率 9600/19200/38400/115200
        :param slave_id: 从站地址 1-31
        :param timeout: 通讯超时时间(秒)
        """
        self.slave_id = slave_id
        self.timeout = timeout
        self.port_name = port
        
        try:
            self.ser = serial.Serial(
                port=port, 
                baudrate=baudrate, 
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=timeout
            )
            logger.info(f"Successfully connected to driver - port: {port}, baudrate: {baudrate}, slave ID: {slave_id}")
            
            # verify connection
            if self._test_connection():
                logger.info("CL57R Driver connection verification successful")
            else:
                logger.warning("CL57R Driver connection verification failed, but serial port is opened")
                
        except serial.SerialException as e:
            logger.error(f"CL57R Driver serial port connection failed: {e}")
            raise DriverError(f"Failed to connect to CL57R Driver: {e}")
    
    def _test_connection(self) -> bool:
        """测试驱动器连接"""
        try:
            # 尝试读取软件版本
            version = self.read_register(0x01, 1)
            if version:
                logger.info(f"CL57R Driver software version: 0x{version[0]:04X}")
                return True
        except Exception as e:
            logger.warning(f"CL57R Driver connection test failed: {e}")
        return False
    
    def _calculate_crc(self, data: bytes) -> bytes:
        """计算CRC-16校验码"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return struct.pack('<H', crc)
    
    def _send_command(self, cmd: bytes) -> Optional[bytes]:
        """发送命令并接收响应"""
        if not self.ser.is_open:  
            logger.error("CL57R Driver serial port is not opened")  
            raise DriverError("CL57R Driver serial port is not opened")  
        
        try:  
            # 计算并添加CRC校验  
            crc = self._calculate_crc(cmd)  
            full_cmd = cmd + crc  
            
            # 清空接收缓冲区  
            self.ser.reset_input_buffer()  
            
            # 发送命令  
            self.ser.write(full_cmd)  
            logger.debug(f"CL57R Driver send command: {' '.join(f'{b:02X}' for b in full_cmd)}")  
            
            # 分步接收响应  
            response = b''  
            
            # 先读取前3个字节（地址+功能码+长度信息）  
            header = self.ser.read(3)  
            if len(header) < 3:  
                logger.error(f"CL57R Driver slave {self.slave_id} has no response")  
                raise DriverError(f"CL57R Driver slave {self.slave_id} communication timeout")  
            
            response += header  
            
            # 检查是否为错误响应  
            if header[1] & 0x80:  
                # 错误响应：再读取2个字节（错误码+CRC）  
                remaining = self.ser.read(2)  
                response += remaining  
            else:  
                # 正常响应：根据功能码确定剩余长度  
                func_code = header[1]  
                if func_code == 0x03:  # 读取响应  
                    data_length = header[2]  # 数据字节数  
                    remaining = self.ser.read(data_length + 2)  # 数据 + CRC(2字节)  
                elif func_code in [0x06, 0x10]:  # 写入响应  
                    remaining = self.ser.read(5)  # 地址(2) + 数据/数量(2) + CRC(2)  
                else:  
                    remaining = self.ser.read(5)  # 默认长度  
                
                response += remaining  
            
            if not response:  
                logger.error(f"CL57R Driver slave {self.slave_id} has no response")  
                raise DriverError(f"CL57R Driver slave {self.slave_id} communication timeout")
            
            logger.debug(f"CL57R Driver receive response: {' '.join(f'{b:02X}' for b in response)}")
            
            # 检查错误响应
            if len(response) >= 3 and (response[1] & 0x80):
                error_code = response[2]
                error_msg = self._get_error_message(error_code)
                logger.error(f"CL57R Driver return error: {error_msg} (code: 0x{error_code:02X})")
                raise DriverError(f"CL57R Driver error: {error_msg}")
            
            # 验证CRC
            if len(response) >= 5:
                data_part = response[:-2]
                received_crc = response[-2:]
                calculated_crc = self._calculate_crc(data_part)
                
                if received_crc != calculated_crc:
                    logger.error("CL57R Driver CRC verification failed")
                    raise DriverError("CL57R Driver communication data verification failed")
            
            return response
            
        except serial.SerialException as e:
            logger.error(f"CL57R Driver serial port communication error: {e}")
            raise DriverError(f"CL57R Driver serial port communication error: {e}")
    
    def _get_error_message(self, error_code: int) -> str:
        """获取错误信息"""
        error_messages = {
            0x01: "错误功能码或CRC校验码错误",
            0x02: "错误的访问地址", 
            0x03: "错误的数据值或超出范围",
            0x04: "拒绝执行命令"
        }
        return error_messages.get(error_code, f"Unknown error code: 0x{error_code:02X}")
    
    def read_register(self, address: int, count: int = 1) -> Optional[List[int]]:
        """
        读取寄存器 (功能码 0x03)
        :param address: 寄存器地址
        :param count: 读取寄存器数量
        :return: 读取的数据列表
        """
        try:
            cmd = struct.pack('>BBHH', self.slave_id, 0x03, address, count)
            response = self._send_command(cmd)
            
            if response and len(response) >= 5:
                data_length = response[2]
                if len(response) >= 3 + data_length + 2:  # 检查数据完整性
                    data = response[3:3+data_length]
                    values = list(struct.unpack('>' + 'H' * (data_length // 2), data))
                    logger.debug(f"CL57R Driver read register 0x{address:04X} successful, count: {count}, values: {values}")
                    return values
                else:
                    logger.error("CL57R Driver receive data length is not enough")
                    
        except Exception as e:
            logger.error(f"CL57R Driver read register 0x{address:04X} failed: {e}")
            raise DriverError(f"CL57R Driver read register failed: {e}")
        
        return None
    
    def write_single_register(self, address: int, value: int) -> bool:
        """
        写单个寄存器 (功能码 0x06)
        :param address: 寄存器地址
        :param value: 写入值
        :return: 是否成功
        """
        try:
            # 检查值的范围
            if not (0 <= value <= 0xFFFF):
                logger.error(f"CL57R Driver write value {value} out of range [0, 65535]")
                raise DriverError(f"CL57R Driver write value out of range: {value}")
                
            cmd = struct.pack('>BBHH', self.slave_id, 0x06, address, value)
            response = self._send_command(cmd)
            
            if response and len(response) >= 8:
                # 验证回显数据
                recv_addr = struct.unpack('>H', response[2:4])[0]
                recv_value = struct.unpack('>H', response[4:6])[0]
                
                if recv_addr == address and recv_value == value:
                    logger.debug(f"CL57R Driver write register 0x{address:04X} = {value} successful")
                    return True
                else:
                    logger.error(f"CL57R Driver write register confirmation failed - address: 0x{recv_addr:04X}, value: {recv_value}")
                    
        except Exception as e:
            logger.error(f"CL57R Driver write register 0x{address:04X} = {value} failed: {e}")
            raise DriverError(f"CL57R Driver write register failed: {e}")
            
        return False
    
    def write_multiple_registers(self, start_address: int, values: List[int]) -> bool:
        """
        写多个寄存器 (功能码 0x10)
        :param start_address: 起始地址
        :param values: 值列表
        :return: 是否成功
        """
        try:
            if not values:
                logger.error("CL57R Driver write value list is empty")
                return False
                
            # 检查值的范围
            for i, value in enumerate(values):
                if not (0 <= value <= 0xFFFF):
                    logger.error(f"CL57R Driver write value {value} out of range [0, 65535]")
                    raise DriverError(f"CL57R Driver write value out of range: index {i}, value {value}")
            
            count = len(values)
            byte_count = count * 2
            cmd = struct.pack('>BBHHB', self.slave_id, 0x10, start_address, count, byte_count)
            
            for value in values:
                cmd += struct.pack('>H', value)
            
            response = self._send_command(cmd)
            
            if response and len(response) >= 8:
                recv_addr = struct.unpack('>H', response[2:4])[0]
                recv_count = struct.unpack('>H', response[4:6])[0]
                
                if recv_addr == start_address and recv_count == count:
                    logger.debug(f"CL57R Driver write multiple registers successful - start address: 0x{start_address:04X}, count: {count}")
                    return True
                else:
                    logger.error(f"CL57R Driver write multiple registers confirmation failed - address: 0x{recv_addr:04X}, count: {recv_count}")
                    
        except Exception as e:
            logger.error(f"CL57R Driver write multiple registers failed: {e}")
            raise DriverError(f"CL57R Driver write multiple registers failed: {e}")
            
        return False
    
    # 32位数据处理辅助函数
    def _write_32bit_value(self, high_addr: int, value: int) -> bool:
        """写入32位值到连续的两个16位寄存器"""
        if not (-2147483648 <= value <= 2147483647):
            logger.error(f"CL57R Driver write 32-bit value {value} out of range [-2147483648, 2147483647]")
            raise DriverError(f"CL57R Driver write 32-bit value out of range: {value}")
            
        # 处理负数
        if value < 0:
            value = value & 0xFFFFFFFF
            
        high = (value >> 16) & 0xFFFF
        low = value & 0xFFFF
        
        return self.write_multiple_registers(high_addr, [high, low])
    
    def _read_32bit_value(self, high_addr: int) -> Optional[int]:
        """从连续的两个16位寄存器读取32位值"""
        data = self.read_register(high_addr, 2)
        if data and len(data) == 2:
            # 组合32位值
            value = (data[0] << 16) | data[1]
            # 处理有符号数
            if value & 0x80000000:
                value = value - 0x100000000
            return value
        return None
    
    # 位置控制相关方法
    def set_position_target(self, position: int) -> bool:
        """
        设置定位目标位置
        :param position: 目标位置 (脉冲数, -2147483648 ~ 2147483647)
        """
        logger.info(f"CL57R Driver set position target: {position} pulse")
        return self._write_32bit_value(0x37, position)
    
    def set_position_speed(self, speed: int) -> bool:
        """
        设置定位运行速度
        :param speed: 速度 (r/min, 0-3000)
        """
        if not (0 <= speed <= 3000):
            logger.error(f"CL57R Driver set position speed {speed} out of range [0, 3000] r/min")
            raise DriverError(f"CL57R Driver set position speed out of range: {speed}")
            
        logger.info(f"CL57R Driver set position speed: {speed} r/min")
        return self.write_single_register(0x36, speed)
    
    def set_position_acceleration_time(self, accel_time: int) -> bool:
        """设置定位加速时间"""
        if not (0 <= accel_time <= 2000):
            logger.error(f"CL57R Driver set position acceleration time {accel_time} out of range [0, 2000] ms")
            raise DriverError(f"CL57R Driver set position acceleration time out of range: {accel_time}")
            
        logger.info(f"CL57R Driver set position acceleration time: {accel_time} ms")
        return self.write_single_register(0x34, accel_time)
    
    def set_position_deceleration_time(self, decel_time: int) -> bool:
        """设置定位减速时间"""
        if not (0 <= decel_time <= 2000):
            logger.error(f"CL57R Driver set position deceleration time {decel_time} out of range [0, 2000] ms")
            raise DriverError(f"CL57R Driver set position deceleration time out of range: {decel_time}")
            
        logger.info(f"CL57R Driver set position deceleration time: {decel_time} ms")
        return self.write_single_register(0x35, decel_time)
    
    def start_relative_positioning(self) -> bool:
        """启动相对定位运动"""
        logger.info("CL57R Driver start relative positioning")
        return self.write_single_register(0x4E, 0x01)  # Bit0=1
    
    def start_absolute_positioning(self) -> bool:
        """启动绝对定位运动"""
        logger.info("CL57R Driver start absolute positioning")
        return self.write_single_register(0x4E, 0x03)  # Bit0=1, Bit1=1
    
    def stop_motion(self) -> bool:
        """停止运动"""
        logger.info("CL57R Driver stop motion")
        return self.write_single_register(0x4E, 0x20)  # Bit5=1
    
    def emergency_stop(self) -> bool:
        """急停"""
        logger.warning("CL57R Driver emergency stop")
        return self.write_single_register(0x4E, 0x40)  # Bit6=1
    
    # JOG控制
    def set_jog_speed(self, speed: int) -> bool:
        """设置JOG速度"""
        if not (-3000 <= speed <= 3000):
            logger.error(f"CL57R Driver set jog speed {speed} out of range [-3000, 3000] r/min")
            raise DriverError(f"CL57R Driver set jog speed out of range: {speed}")
            
        logger.info(f"CL57R Driver set jog speed: {speed} r/min")
        return self.write_single_register(0x30, speed)
    
    def start_jog(self, speed: int, direction: int = 1) -> bool:
        """
        启动JOG运动
        :param speed: JOG速度 (r/min, 0-3000)
        :param direction: 方向 1=正向, -1=负向
        """
        actual_speed = speed if direction > 0 else -speed
        logger.info(f"CL57R Driver start jog - speed: {actual_speed} r/min")
        
        if self.set_jog_speed(actual_speed):
            return self.write_single_register(0x4E, 0x08)  # Bit3=1
        return False
    
    # 回零控制
    def set_home_mode(self, mode: HomeMode) -> bool:
        """设置回零方式"""
        logger.info(f"CL57R Driver set home mode: {mode.name} (value: {mode.value})")
        return self.write_single_register(0x40, mode.value)
    
    def set_home_speed(self, speed: int) -> bool:
        """设置回零速度"""
        if not (0 <= speed <= 3000):
            logger.error(f"CL57R Driver set home speed {speed} out of range [0, 3000] r/min")
            raise DriverError(f"CL57R Driver set home speed out of range: {speed}")
            
        logger.info(f"CL57R Driver set home speed: {speed} r/min")
        return self.write_single_register(0x41, speed)
    
    def set_home_creep_speed(self, speed: int) -> bool:
        """设置回零爬行速度"""
        if not (0 <= speed <= 3000):
            logger.error(f"CL57R Driver set home creep speed {speed} out of range [0, 3000] r/min")
            raise DriverError(f"CL57R Driver set home creep speed out of range: {speed}")
            
        logger.info(f"CL57R Driver set home creep speed: {speed} r/min")
        return self.write_single_register(0x42, speed)
    
    def start_homing(self, mode: HomeMode = HomeMode.POSITIVE_HOME) -> bool:
        """
        启动回零
        :param mode: 回零方式
        """
        logger.info(f"CL57R Driver start homing - mode: {mode.name}")
        if self.set_home_mode(mode):
            return self.write_single_register(0x4E, 0x10)  # Bit4=1
        return False
    
    # 状态监控
    def get_current_position(self) -> Optional[int]:
        """获取当前位置"""
        position = self._read_32bit_value(0x08)
        if position is not None:
            logger.debug(f"CL57R Driver current position: {position} pulse")
        return position
    
    def get_current_speed(self) -> Optional[int]:
        """获取当前速度"""
        data = self.read_register(0x0A, 1)
        if data:
            speed = data[0]
            # 处理有符号数
            if speed & 0x8000:
                speed = speed - 0x10000
            logger.debug(f"CL57R Driver current speed: {speed} r/min")
            return speed
        return None
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """获取运行状态"""
        try:
            data = self.read_register(0x04, 1)
            if data:
                status = data[0]
                status_dict = {
                    'in_position': bool(status & 0x01),      # Bit0: 到位
                    'homing_done': bool(status & 0x02),      # Bit1: 回零完成
                    'motor_running': bool(status & 0x04),    # Bit2: 电机运行
                    'fault': bool(status & 0x08),            # Bit3: 故障
                    'motor_enabled': bool(status & 0x10),    # Bit4: 电机使能
                    'positive_limit': bool(status & 0x20),   # Bit5: 正软限位
                    'negative_limit': bool(status & 0x40),   # Bit6: 负软限位
                    'raw_value': status
                }
                
                # 打印状态变化
                if hasattr(self, '_last_status'):
                    for key, value in status_dict.items():
                        if key != 'raw_value' and self._last_status.get(key) != value and value:
                            logger.info(f"CL57R Driver status change: {key} = {value}")
                
                self._last_status = status_dict.copy()
                return status_dict
                
        except Exception as e:
            logger.error(f"CL57R Driver get status failed: {e}")
            
        return None
    
    def get_alarm_status(self) -> Optional[Dict[str, Any]]:
        """获取报警状态"""
        try:
            data = self.read_register(0x05, 1)
            if data:
                alarm_code = data[0]
                alarm_info = {
                    'alarm_code': alarm_code,
                    'has_alarm': alarm_code != 0,
                    'alarm_name': None,
                    'alarm_description': None
                }
                
                if alarm_code != 0:
                    try:
                        alarm_enum = AlarmCode(alarm_code)
                        alarm_info['alarm_name'] = alarm_enum.name
                        
                        alarm_descriptions = {
                            AlarmCode.OVERCURRENT: "过流报警 - 检查电机线路和负载",
                            AlarmCode.OVERVOLTAGE: "过压报警 - 检查供电电压",
                            AlarmCode.UNDERVOLTAGE: "欠压报警 - 检查供电电压",
                            AlarmCode.MEMORY_ERROR: "存储器错误 - 驱动器故障",
                            AlarmCode.POSITION_ERROR: "位置超差 - 检查机械负载和参数"
                        }
                        alarm_info['alarm_description'] = alarm_descriptions.get(alarm_enum, "未知报警")
                        
                        logger.warning(f"CL57R Driver alarm: {alarm_info['alarm_name']} - {alarm_info['alarm_description']}")
                        
                    except ValueError:
                        alarm_info['alarm_name'] = f"UNKNOWN_0x{alarm_code:02X}"
                        alarm_info['alarm_description'] = f"未知报警代码: 0x{alarm_code:02X}"
                        logger.warning(f"未知报警代码: 0x{alarm_code:02X}")
                
                return alarm_info
                
        except Exception as e:
            logger.error(f"CL57R Driver get alarm status failed: {e}")
            
        return None
    
    def clear_alarm(self) -> bool:
        """清除当前报警"""
        logger.info("清除报警")
        return self.write_single_register(0x4F, 0x0300)
    
    def get_di_status(self) -> Optional[Dict[str, bool]]:
        """获取DI端口状态"""
        try:
            data = self.read_register(0x06, 1)
            if data:
                di_status = data[0]
                status_dict = {}
                for i in range(7):  # DI0-DI6
                    status_dict[f'DI{i}'] = bool(di_status & (1 << i))
                    
                logger.debug(f"CL57R Driver DI status: {status_dict}")
                return status_dict
                
        except Exception as e:
            logger.error(f"CL57R Driver get DI status failed: {e}")
            
        return None
    
    def get_do_status(self) -> Optional[Dict[str, bool]]:
        """获取DO端口状态"""
        try:
            data = self.read_register(0x07, 1)
            if data:
                do_status = data[0]
                status_dict = {}
                for i in range(3):  # DO0-DO2
                    status_dict[f'DO{i}'] = bool(do_status & (1 << i))
                    
                logger.debug(f"CL57R Driver DO status: {status_dict}")
                return status_dict
                
        except Exception as e:
            logger.error(f"CL57R Driver get DO status failed: {e}")
            
        return None
    
    # DI/DO配置
    def configure_di_function(self, di_port: int, function: int) -> bool:
        """
        配置DI端口功能
        :param di_port: DI端口号 0-6
        :param function: 功能码 (参考手册DI功能命令表)
        """
        if not (0 <= di_port <= 6):
            logger.error(f"CL57R Driver configure DI port {di_port} out of range [0, 6]")
            raise DriverError(f"CL57R Driver configure DI port out of range: {di_port}")
            
        if not (0 <= function <= 17):
            logger.error(f"CL57R Driver configure DI function {function} out of range [0, 17]")
            raise DriverError(f"CL57R Driver configure DI function out of range: {function}")
            
        address = 0x11 + di_port  # PA_011 to PA_017
        logger.info(f"CL57R Driver configure DI{di_port} function: {function}")
        return self.write_single_register(address, function)
    
    def configure_do_function(self, do_port: int, function: int) -> bool:
        """
        配置DO端口功能
        :param do_port: DO端口号 0-2
        :param function: 功能码 (参考手册DO功能命令表)
        """
        if not (0 <= do_port <= 2):
            logger.error(f"CL57R Driver configure DO port {do_port} out of range [0, 2]")
            raise DriverError(f"CL57R Driver configure DO port out of range: {do_port}")
            
        if not (0 <= function <= 11):
            logger.error(f"CL57R Driver configure DO function {function} out of range [0, 11]")
            raise DriverError(f"CL57R Driver configure DO function out of range: {function}")
            
        address = 0x1C + do_port  # PA_01C to PA_01E
        logger.info(f"CL57R Driver configure DO{do_port} function: {function}")
        return self.write_single_register(address, function)
    
    def motor_enable(self) -> bool:
        """电机使能"""
        logger.info("CL57R Driver motor enable")
        return self.write_single_register(0x4F, 0x0500)
    
    def motor_disable(self) -> bool:
        """电机释放"""
        logger.info("CL57R Driver motor disable")
        return self.write_single_register(0x4F, 0x0600)
    
    def save_parameters(self) -> bool:
        """保存当前参数"""
        logger.info("CL57R Driver save parameters")
        return self.write_single_register(0x4F, 0x0200)
    
    def restore_factory_settings(self) -> bool:
        """恢复出厂设置"""
        logger.warning("CL57R Driver restore factory settings")
        return self.write_single_register(0x4F, 0x0100)
    
    def clear_position(self) -> bool:
        """清除当前位置"""
        logger.info("CL57R Driver clear position")
        return self.write_single_register(0x4F, 0x0400)
    
    def wait_for_in_position(self, timeout: float = 30.0, check_interval: float = 0.1) -> bool:
        """
        等待电机到位
        :param timeout: 超时时间(秒)
        :param check_interval: 检查间隔(秒)
        :return: 是否到位
        """
        logger.info(f"CL57R Driver wait for motor in position (timeout: {timeout}s)")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                status = self.get_status()
                if status:
                    if status['fault']:
                        logger.error("CL57R Driver motor running fault")
                        return False
                        
                    if status['in_position']:
                        logger.info("CL57R Driver motor in position")
                        return True
                        
                    # 显示进度信息
                    if hasattr(self, '_last_position_log') and (time.time() - self._last_position_log) > 1.0:
                        position = self.get_current_position()
                        speed = self.get_current_speed()
                        logger.info(f"CL57R Driver moving - position: {position}, speed: {speed} r/min")
                        self._last_position_log = time.time()
                    elif not hasattr(self, '_last_position_log'):
                        self._last_position_log = time.time()
                        
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"CL57R Driver wait for in position failed: {e}")
                return False
        
        logger.warning(f"CL57R Driver wait for in position timeout ({timeout}s)")
        return False
    
    def wait_for_homing_complete(self, timeout: float = 60.0, check_interval: float = 0.1) -> bool:
        """
        等待回零完成
        :param timeout: 超时时间(秒) 
        :param check_interval: 检查间隔(秒)
        :return: 是否回零完成
        """
        logger.info(f"CL57R Driver wait for homing complete (timeout: {timeout}s)")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                status = self.get_status()
                if status:
                    if status['fault']:
                        logger.error("CL57R Driver homing fault")
                        return False
                        
                    if status['homing_done']:
                        logger.info("CL57R Driver homing complete")
                        return True
                        
                    # 显示进度信息
                    if hasattr(self, '_last_home_log') and (time.time() - self._last_home_log) > 2.0:
                        position = self.get_current_position()
                        speed = self.get_current_speed()
                        logger.info(f"CL57R Driver homing - position: {position}, speed: {speed} r/min")
                        self._last_home_log = time.time()
                    elif not hasattr(self, '_last_home_log'):
                        self._last_home_log = time.time()
                        
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"CL57R Driver wait for homing complete failed: {e}")
                return False
        
        logger.warning(f"CL57R Driver wait for homing complete timeout ({timeout}s)")
        return False
    
    def print_status_info(self) -> None:
        """打印详细状态信息"""
        try:
            print("\n" + "="*50)
            print("驱动器状态信息")
            print("="*50)
            
            # 基本状态
            status = self.get_status()
            if status:
                print("运行状态:")
                for key, value in status.items():
                    if key != 'raw_value':
                        print(f"  {key}: {value}")
            
            # 位置和速度
            position = self.get_current_position()
            speed = self.get_current_speed()
            print(f"\n当前位置: {position} 脉冲")
            print(f"当前速度: {speed} r/min")
            
            # 报警状态
            alarm = self.get_alarm_status()
            if alarm and alarm['has_alarm']:
                print(f"\n⚠️  报警信息:")
                print(f"  代码: 0x{alarm['alarm_code']:02X}")
                print(f"  名称: {alarm['alarm_name']}")
                print(f"  描述: {alarm['alarm_description']}")
            else:
                print("\n✅ 无报警")
            
            # DI/DO状态
            di_status = self.get_di_status()
            do_status = self.get_do_status()
            
            if di_status:
                print(f"\nDI状态: {di_status}")
            if do_status:
                print(f"DO状态: {do_status}")
                
            print("="*50)
            
        except Exception as e:
            logger.error(f"打印状态信息失败: {e}")
    
    def close(self) -> None:
        """关闭串口连接"""
        try:
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
                logger.info(f"CL57R Driver close serial port: {self.port_name}")
        except Exception as e:
            logger.error(f"CL57R Driver close serial port failed: {e}")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        self.close()

