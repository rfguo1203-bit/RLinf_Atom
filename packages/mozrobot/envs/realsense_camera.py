import logging
from typing import Optional, Callable, Dict
import enum
import time
import signal
import os
import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from mozrobot.envs.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from mozrobot.envs.shared_memory.shared_ndarray import SharedNDArray
from mozrobot.envs.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from mozrobot.envs.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
# import pyzed.sl as sl

import pyrealsense2 as rs

class CameraError(Exception):
    """Exception raised when camera encounters unrecoverable errors."""
    pass

class RealSenseCamera(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            serial_number,
            camera_name: str,
            resolution=(640, 480),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            enable_color=True,
            enable_depth=False,
            enable_point_cloud=False,
            enable_tracking=False,
            get_max_k=30,
            receive_latency=0.0,
            num_threads=2,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            verbose=False,
            max_consecutive_errors=1,  # Maximum consecutive errors before marking as failed
            simu_mode=False,
        ):
        super().__init__(name=camera_name)

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['left_color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
            examples['right_color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.float32)
        if enable_point_cloud:
            examples['point_cloud'] = np.empty(
                shape=shape+(4,), dtype=np.float32)    # The last float is used to store color information, where R, G, B, and alpha channels (4 x 8-bit) are concatenated into a single 32-bit float.
        if enable_tracking:
            examples['camera_pose'] = np.zeros((7,), dtype=np.float32)   # x, y, z, qx, qy, qz, qw
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=2.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=2.2,
            put_desired_frequency=put_fps
        )

        # create shared array for intrinsics
        left_intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(6,),
                dtype=np.float64)
        right_intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(6,),
                dtype=np.float64)
        left_intrinsics_array.get()[:] = 0
        right_intrinsics_array.get()[:] = 0

        # create shared error state variables
        error_state_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(2,),  # [has_error, consecutive_error_count]
                dtype=np.int32)
        error_state_array.get()[:] = 0

        # copied variables
        self.shm_manager = shm_manager
        self.serial_number = serial_number
        self.camera_name = camera_name
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_point_cloud = enable_point_cloud
        self.enable_tracking = enable_tracking
        self.receive_latency = receive_latency
        self.transform = transform
        self.vis_transform = vis_transform
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads
        self.max_consecutive_errors = max_consecutive_errors
        self.simu_mode = simu_mode

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.left_intrinsics_array = left_intrinsics_array
        self.right_intrinsics_array = right_intrinsics_array
        self.error_state_array = error_state_array

    @staticmethod
    def get_connected_devices_serial(realsense_config, simu_mode=False):
        """
        获取连接的摄像头序列号列表，支持自动识别：
        1. USB端口配置 (包含"-"的格式，如 "2-1,2-3,1-4")
        2. 直接序列号配置 (纯数字或字母数字，如 "123456,789012,345678")

        :param realsense_config: 相机配置字符串，自动识别类型
        :param simu_mode: 仿真模式
        :return: 序列号列表
        """
        if simu_mode:
            return ["123", "456", "789"]

        if realsense_config is None:
            raise ValueError("realsense_config cannot be None when not in simu_mode")

        realsense_serial_numbers = []
        for part in realsense_config:
            if isinstance(part, str):
                part = part.strip()

            if isinstance(part, str) and "-" in part:
                logging.debug(f"Detected USB port configuration: {part}")
                realsense_serial_numbers.extend(RealSenseCamera._get_serials_by_usb_ports(part))
            else:
                logging.debug(f"Detected direct serial number configuration: {part}")
                realsense_serial_numbers.append(part)

        return realsense_serial_numbers

    @staticmethod
    def _get_serials_by_usb_ports(usb_ports_config):
        """通过USB端口匹配序列号 - 仅在配置初始化时使用"""
        if isinstance(usb_ports_config, str):
            sorted_usb_ports = [p.strip() for p in usb_ports_config.split(",")]
        else:
            sorted_usb_ports = usb_ports_config
        ctx = rs.context()
        devices = ctx.query_devices()
        logging.debug(f"devices: {devices}")

        # 获取摄像头序列号与其 USB 通道号的映射
        cameras = []
        for device in devices:
            logging.debug(f"device: {device}")
            serial_number = device.get_info(rs.camera_info.serial_number)
            physical_port = device.get_info(rs.camera_info.physical_port)

            # 记录 USB 通道号
            cameras.append((physical_port, serial_number))
            logging.debug(f"Device Serial: {serial_number}, Physical Port: {physical_port}")

        # 如果未指定排序的 USB 通道号，返回默认未排序的序列号
        if not sorted_usb_ports:
            return [serial for _, serial in cameras]

        # 按用户指定的 USB 通道号查找匹配的序列号
        matched_cameras = []
        for usb_port in sorted_usb_ports:
            for physical_port, serial in cameras:
                # 检查物理端口是否包含指定的 USB 通道号
                if usb_port in physical_port:
                    matched_cameras.append(serial)
                    logging.debug(f"Match Found: {serial} for USB Port: {usb_port}")

        logging.info(f"Return matched serial SN: {matched_cameras}")
        return matched_cameras

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, cx, cy = self.left_intrinsics_array.get()[:4]
        left_mat = np.eye(3)
        left_mat[0, 0] = fx
        left_mat[1, 1] = fy
        left_mat[0, 2] = cx
        left_mat[1, 2] = cy
        fx, fy, cx, cy = self.right_intrinsics_array.get()[:4]
        right_mat = np.eye(3)
        right_mat[0, 0] = fx
        right_mat[1, 1] = fy
        right_mat[0, 2] = cx
        right_mat[1, 2] = cy
        return left_mat

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self, timeout=10.0):
        if (self.simu_mode):
            return True  # Simulation mode always succeeds

        # Wait for ready event with timeout
        if not self.ready_event.wait(timeout=timeout):
            logging.error(f"RealSense camera {self.camera_name} failed to initialize within {timeout}s timeout")
            # Mark as failed if initialization timed out
            if hasattr(self, 'error_state_array'):
                try:
                    error_state = self.error_state_array.get()
                    error_state[0] = 1  # has_error = True
                    error_state[1] = 1  # consecutive_error_count = 1
                except:
                    pass  # Ignore errors accessing shared memory

            # Force set ready event to unblock other waiting code
            self.ready_event.set()

            # If process is still alive but not responding, terminate it
            if self.is_alive():
                logging.warning(f"Terminating unresponsive camera process {self.camera_name}")
                self.terminate()
                self.join(timeout=2)
                # Force kill if still alive
                if self.is_alive():
                    logging.error(f"Force killing unresponsive camera process {self.camera_name}")
                    self.kill()
            return False

        # Check if camera failed during initialization
        if hasattr(self, 'error_state_array'):
            try:
                error_state = self.error_state_array.get()
                if error_state[0]:  # has_error is True
                    error_count = error_state[1] if len(error_state) > 1 else 1
                    logging.error(f"RealSense camera {self.camera_name} failed during initialization (error count: {error_count})")
                    return False
            except Exception as e:
                logging.warning(f"Could not check error state for camera {self.camera_name}: {e}")
                # If we can't check error state and the process is still running, it might be stuck
                if self.is_alive():
                    logging.warning(f"Camera {self.camera_name} process alive but error state inaccessible - assuming failure")
                    return False

        # Additional validation: check if process is alive and responsive
        if not self.is_alive():
            logging.error(f"RealSense camera {self.camera_name} process died during initialization")
            return False

        logging.info(f"RealSense camera {self.camera_name} initialized successfully")
        return True

    def end_wait(self):
        self.join(timeout=5.0)  # Add timeout to prevent hanging
        if self.is_alive():
            logging.warning(f"Force terminating camera process {self.camera_name}")
            self.terminate()
            self.join(timeout=2.0)
            if self.is_alive():
                logging.error(f"Force killing camera process {self.camera_name}")
                self.kill()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        # Check for camera errors before returning data
        error_state = self.error_state_array.get()
        if error_state[0]:  # has_error is True
            consecutive_errors = error_state[1]
            raise CameraError(
                f"RealSense camera {self.serial_number} encountered {consecutive_errors} consecutive errors. "
                f"Camera process may have failed."
            )

        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        # Check for camera errors before returning data
        error_state = self.error_state_array.get()
        if error_state[0]:  # has_error is True
            consecutive_errors = error_state[1]
            raise CameraError(
                f"RealSense camera {self.serial_number} encountered {consecutive_errors} consecutive errors. "
                f"Camera process may have failed."
            )

        return self.vis_ring_buffer.get(out=out)

    def get_error_status(self):
        """Get current error status of the camera.

        Returns:
            dict: Dictionary containing error information:
                - has_error: bool, whether camera has failed
                - consecutive_errors: int, number of consecutive errors
        """
        error_state = self.error_state_array.get()
        return {
            'has_error': bool(error_state[0]),
            'consecutive_errors': int(error_state[1])
        }

    # ========= interval API ===========
    def run(self):
        if (self.simu_mode):
            return

        # Set up signal handling for graceful shutdown with time-based force exit
        _shutdown_requested = [False]  # Use list to allow modification in nested function
        _force_exit_time = [None]

        def signal_handler(signum, frame):
            current_time = time.time()

            # Force exit if shutdown was already requested (double Ctrl+C within 2 seconds)
            if _shutdown_requested[0]:
                if _force_exit_time[0] and (current_time - _force_exit_time[0]) < 2.0:
                    logging.warning(f"🚨 Camera {self.camera_name} force exiting (double Ctrl+C)")
                    os._exit(1)
                else:
                    _force_exit_time[0] = current_time
                    logging.warning(f"🚨 Camera {self.camera_name} second Ctrl+C - Press again within 2s to force exit")
                    return

            _shutdown_requested[0] = True
            _force_exit_time[0] = current_time
            logging.info(f"🛑 Camera {self.camera_name} received interrupt signal, shutting down...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # limit threads
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        try:
            # Initialize ZED camera
            self.rsc = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)

            w, h = self.resolution[0], self.resolution[1]
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.capture_fps)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.capture_fps)
            # if self.enable_tracking:
            #     config.enable_stream(rs.stream.pose)

            self.rsc.start(config)
        except Exception as e:
            logging.error(f"Failed to initialize RealSense camera {self.camera_name}: {e}")
            logging.error(f"Camera {self.camera_name} error details: {type(e).__name__}: {str(e)}")

            # Mark camera as failed with proper error handling
            try:
                error_state = self.error_state_array.get()
                error_state[0] = 1  # has_error = True
                error_state[1] = 1  # consecutive_error_count = 1
                logging.debug(f"Camera {self.camera_name} error state set successfully")
            except Exception as state_error:
                logging.error(f"Failed to set error state for camera {self.camera_name}: {state_error}")

            # Always set ready event to unblock waiting threads, even if camera failed
            try:
                self.ready_event.set()
                logging.debug(f"Camera {self.camera_name} ready event set")
            except Exception as ready_error:
                logging.error(f"Failed to set ready event for camera {self.camera_name}: {ready_error}")

            # Set stop event to ensure graceful shutdown
            try:
                self.stop_event.set()
                logging.debug(f"Camera {self.camera_name} stop event set")
            except Exception as stop_error:
                logging.error(f"Failed to set stop event for camera {self.camera_name}: {stop_error}")

            return

        # if self.enable_tracking:
        #     tracking_params = sl.PositionalTrackingParameters()
        #     tracking_params.enable_imu_fusion = True
        #     tracking_params.enable_pose_smoothing = True
        #     err = self.rsc.enable_positional_tracking(tracking_params)
        #     if err != sl.ERROR_CODE.SUCCESS:
        #         raise RuntimeError(f"Failed to enable positional tracking: {repr(err)}")

        color_sensor_intrinsics = self.rsc.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        order = ['fx', 'fy', 'ppx', 'ppy']
        for i, name in enumerate(order):
            self.left_intrinsics_array.get()[i] = getattr(color_sensor_intrinsics, name)
            self.right_intrinsics_array.get()[i] = getattr(color_sensor_intrinsics, name)
        self.left_intrinsics_array.get()[4] = color_sensor_intrinsics.height
        self.left_intrinsics_array.get()[5] = color_sensor_intrinsics.width
        self.right_intrinsics_array.get()[4] = color_sensor_intrinsics.height
        self.right_intrinsics_array.get()[5] = color_sensor_intrinsics.width

        # # Prepare to capture frames from ZED camera
        # runtime_params = sl.RuntimeParameters()
        # # runtime_params.enable_fill_mode = True   # Fill holes and occlusions in the depth map
        # left_image = sl.Mat(w, h, sl.MAT_TYPE.U8_C4)
        # right_image = sl.Mat(w, h, sl.MAT_TYPE.U8_C4)
        # depth = sl.Mat(w, h, sl.MAT_TYPE.F32_C1)
        # point_cloud = sl.Mat()
        # camera_pose = sl.Pose()

        try:
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()
            consecutive_errors = 0
            while not self.stop_event.is_set():
                try:
                    frames = self.rsc.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    # Reset error count on successful frame capture
                    consecutive_errors = 0
                    error_state = self.error_state_array.get()
                    error_state[0] = 0  # has_error = False
                    error_state[1] = 0  # consecutive_error_count = 0

                except KeyboardInterrupt:
                    logging.info(f"RealSense camera {self.camera_name} process interrupted by user")
                    break

                except Exception as e:
                    consecutive_errors += 1
                    logging.error(f"Error getting frames from RealSense camera {self.camera_name}: {e} (consecutive errors: {consecutive_errors})")

                    # Update shared error state
                    error_state = self.error_state_array.get()
                    error_state[1] = consecutive_errors  # consecutive_error_count

                    # Mark as failed if too many consecutive errors
                    if consecutive_errors >= self.max_consecutive_errors:
                        error_state[0] = 1  # has_error = True
                        logging.error(f"RealSense camera {self.camera_name} marked as failed after {consecutive_errors} consecutive errors")

                    continue

                if True:
                    # self.rsc.retrieve_image(left_image, sl.VIEW.LEFT)
                    # self.rsc.retrieve_image(right_image, sl.VIEW.RIGHT)
                    # self.rsc.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    # self.rsc.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                    # if self.enable_tracking:
                    #     tracking_state = self.rsc.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)  # Get the position of the camera in a fixed reference frame (the World Frame)

                    t_recv = time.time()
                    # t_cap = self.rsc.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds() / 1000
                    t_cap = color_frame.get_timestamp()
                    t_cal = t_recv - self.receive_latency  # calibrated latency

                    data = dict()
                    data['camera_receive_timestamp'] = t_recv
                    data['camera_capture_timestamp'] = t_cap
                    if self.enable_color:
                        data['left_color'] = np.asanyarray(color_frame.get_data())[:, :, :3]
                        # data['right_color'] = np.asanyarray(color_frame.get_data())[:, :, :3]
                        data['right_color'] = data['left_color']
                    if self.enable_depth:
                        depth_frame = frames.get_depth_frame()
                        data['depth'] = depth_frame.get_data()
                    # if self.enable_point_cloud:
                    #     data['point_cloud'] = point_cloud.get_data()
                    # if self.enable_tracking:
                    #     if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    #         py_translation = sl.Translation()
                    #         tx = round(camera_pose.get_translation(py_translation).get()[0], 3)
                    #         ty = round(camera_pose.get_translation(py_translation).get()[1], 3)
                    #         tz = round(camera_pose.get_translation(py_translation).get()[2], 3)
                    #         py_orientation = sl.Orientation()
                    #         qx = round(camera_pose.get_orientation(py_orientation).get()[0], 3)
                    #         qy = round(camera_pose.get_orientation(py_orientation).get()[1], 3)
                    #         qz = round(camera_pose.get_orientation(py_orientation).get()[2], 3)
                    #         qw = round(camera_pose.get_orientation(py_orientation).get()[3], 3)
                    #     else:
                    #         tx, ty, tz, qx, qy, qz, qw = float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
                    #     data['camera_pose'] = np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float32)

                    # apply transform
                    put_data = data
                    if self.transform is not None:
                        put_data = self.transform(dict(data))

                    if self.put_downsample:
                        local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                            timestamps=[t_cal],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                        for step_idx in global_idxs:
                            put_data['step_idx'] = step_idx
                            put_data['timestamp'] = t_cal
                            self.ring_buffer.put(put_data, wait=False)
                    else:
                        step_idx = int((t_cal - put_start_time) * self.put_fps)
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data, wait=False)

                    # signal ready
                    if iter_idx == 0:
                        self.ready_event.set()

                    # put to vis
                    vis_data = data
                    if self.vis_transform == self.transform:
                        vis_data = put_data
                    elif self.vis_transform is not None:
                        vis_data = self.vis_transform(dict(data))
                    self.vis_ring_buffer.put(vis_data, wait=False)

                    # perf
                    t_end = time.time()
                    duration = t_end - t_start
                    frequency = np.round(1 / duration, 1)
                    t_start = t_end
                    if self.verbose:
                        logging.debug(f'[RealSenseCamera {self.camera_name}] FPS {frequency}')

                    iter_idx += 1
        finally:
            # When everything done, release the capture
            self.rsc.stop()
