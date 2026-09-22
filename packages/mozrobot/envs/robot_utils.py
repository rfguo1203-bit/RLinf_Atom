# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import time
import sys
import os
import logging
sys.path.insert(0, os.getcwd())
import numpy as np
from mozrobot.envs.realsense_camera import RealSenseCamera, CameraError
from multiprocessing.managers import SharedMemoryManager
import cv2

class ImageRecorder:
    def __init__(
        self,
        camera_resolutions={"cam_high": (320, 240), "cam_left_wrist": (320, 240), "cam_right_wrist": (320, 240)},
        obs_float32=False,
        shm_manager=None,
        realsense_serials=None,  # 相机配置参数(支持串口号或序列号)
        simu_mode=False,
        no_camera=False,
        disabled_cameras=None,  # 禁用的相机列表
    ):

        self.last_camera_high_data = None
        self.last_camera_left_wrist_data = None
        self.last_camera_right_wrist_data = None

        # Handle disabled cameras
        self.disabled_cameras = disabled_cameras or []
        if isinstance(self.disabled_cameras, str):
            self.disabled_cameras = [self.disabled_cameras]

        # Skip camera initialization if no_camera=True or simu_mode=True
        self.skip_camera = no_camera or simu_mode
        if self.skip_camera:
            reason = "no_camera=True" if no_camera else "simu_mode=True"
            logging.debug(f"Camera functionality disabled ({reason})")
            self.camera_high = None
            self.camera_left_wrist = None
            self.camera_right_wrist = None
            return

        # Log which cameras are disabled
        if self.disabled_cameras:
            logging.info(f"Disabled cameras: {self.disabled_cameras}")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        self.fps = 30
        # all realsense camera using 640x480 resolution
        camera_raw_res = { "cam_high": (640, 480), "cam_left_wrist": (640, 480), "cam_right_wrist": (640, 480)}

        # todo: add depth image
        img_type_list = ['left_color','right_color']
        transform_type_list = [cv2.INTER_AREA, cv2.INTER_AREA]

        def rgb_image_tf(data, input_res, output_res):
            for im_type, tsf_type in zip(img_type_list,transform_type_list):
                img = data[im_type]
                iw, ih = input_res
                ow, oh = output_res
                assert img.shape == ((ih, iw, 3)) or img.shape == ((ih, iw))
                # resize
                img = cv2.resize(img, (ow, oh), interpolation=tsf_type)
                if im_type.split('_')[-1] == 'color':
                    img = np.ascontiguousarray(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                data[im_type] = img
            return data

        rgb_camera_tf_list = {camera_name: lambda data, input_res=raw_res, output_res=transformed_res: rgb_image_tf(data, input_res, output_res)
            for camera_name, raw_res, transformed_res in zip(camera_raw_res.keys(), camera_raw_res.values(), camera_resolutions.values())}

        rs_serials = RealSenseCamera.get_connected_devices_serial(
            realsense_config=realsense_serials,
            simu_mode=simu_mode
        )
        logging.info(f'RealSenseCamera serials: {rs_serials}')

        # Initialize cam_high camera
        if "cam_high" not in self.disabled_cameras:
            logging.debug("camera high init")
            camera_high = RealSenseCamera(
                    shm_manager=shm_manager,
                    serial_number=rs_serials[0],
                    camera_name="cam_high",
                    resolution=camera_raw_res["cam_high"],
                    capture_fps=self.fps,
                    enable_color=True,
                    enable_depth=False,
                    enable_point_cloud=False,
                    enable_tracking=False,
                    transform=rgb_camera_tf_list["cam_high"],
                    put_downsample=True,
                    verbose=False,
                    simu_mode=simu_mode
            )
        else:
            logging.info("Camera cam_high disabled, skipping initialization")
            camera_high = None

        # Initialize cam_left_wrist camera
        if "cam_left_wrist" not in self.disabled_cameras:
            logging.debug("camera left wrist init")
            camera_left_wrist = RealSenseCamera(
                shm_manager=shm_manager,
                serial_number=rs_serials[1],
                camera_name="cam_left_wrist",
                resolution=camera_raw_res["cam_left_wrist"],
                capture_fps=self.fps,
                enable_color=True,
                enable_depth=False,
                enable_point_cloud=False,
                enable_tracking=False,
                transform=rgb_camera_tf_list["cam_left_wrist"],
                put_downsample=True,
                verbose=False,
                simu_mode=simu_mode
            )
        else:
            logging.info("Camera cam_left_wrist disabled, skipping initialization")
            camera_left_wrist = None

        # Initialize cam_right_wrist camera
        if "cam_right_wrist" not in self.disabled_cameras:
            logging.debug("camera right wrist init")
            camera_right_wrist = RealSenseCamera(
                shm_manager=shm_manager,
                serial_number=rs_serials[2],
                camera_name="cam_right_wrist",
                resolution=camera_raw_res["cam_right_wrist"],
                capture_fps=self.fps,
                enable_color=True,
                enable_depth=False,
                enable_point_cloud=False,
                enable_tracking=False,
                transform=rgb_camera_tf_list["cam_right_wrist"],
                put_downsample=True,
                verbose=False,
                simu_mode=simu_mode
            )
        else:
            logging.info("Camera cam_right_wrist disabled, skipping initialization")
            camera_right_wrist = None

        self.camera_high  = camera_high
        self.camera_left_wrist = camera_left_wrist
        self.camera_right_wrist = camera_right_wrist

        time.sleep(1)
        logging.info("All cameras initialized")

    # ======== start-stop API =============
    @property
    def is_ready(self):
        if self.skip_camera:
            return True

        # Check readiness of all enabled cameras
        enabled_cameras = [
            camera for camera in [self.camera_high, self.camera_left_wrist, self.camera_right_wrist]
            if camera is not None
        ]

        # If all cameras are disabled, we still need at least one to be functional
        # unless we're in no_camera or simu_mode (handled by skip_camera above)
        if not enabled_cameras:
            logging.warning("All cameras are disabled but camera system is required")
            return False

        return all(camera.is_ready for camera in enabled_cameras)

    def start(self, wait=True):
        logging.info("Camera system starting...")
        if self.skip_camera:
            logging.info("Camera functionality disabled, skipping camera start")
            return

        # Start only enabled cameras
        if self.camera_high is not None:
            self.camera_high.start(wait=False)
        if self.camera_left_wrist is not None:
            self.camera_left_wrist.start(wait=False)
        if self.camera_right_wrist is not None:
            self.camera_right_wrist.start(wait=False)

        if wait:
            try:
                self.start_wait()
                logging.info("Camera system started successfully")
            except CameraError as e:
                logging.error(f"Camera system startup failed: {e}")
                # Stop any cameras that may have started
                try:
                    self.stop(wait=True)  # Wait for cleanup to ensure proper shutdown
                except Exception as cleanup_error:
                    logging.warning(f"Error during camera cleanup: {cleanup_error}")
                # Re-raise the exception to stop program execution
                raise
        else:
            logging.info("Camera system started (background initialization)")

    def stop(self, wait=True):
        if self.skip_camera:
            logging.info("Camera functionality disabled, skipping camera stop")
            return

        logging.info("Stopping camera system...")

        # Stop all cameras
        cameras = [
            ("cam_high", self.camera_high),
            ("cam_left_wrist", self.camera_left_wrist),
            ("cam_right_wrist", self.camera_right_wrist)
        ]

        for camera_name, camera in cameras:
            if camera is not None:
                try:
                    camera.stop(wait=False)
                except Exception as e:
                    logging.warning(f"Error stopping camera {camera_name}: {e}")

        if wait:
            self.stop_wait()

    def start_wait(self):
        if self.skip_camera:
            return

        # Check each camera startup and collect failures
        failed_cameras = []
        cameras = [
            ("cam_high", self.camera_high),
            ("cam_left_wrist", self.camera_left_wrist),
            ("cam_right_wrist", self.camera_right_wrist)
        ]

        for camera_name, camera in cameras:
            if camera is not None:  # Skip disabled cameras
                success = camera.start_wait()
                if not success:
                    failed_cameras.append(camera_name)
                    logging.error(f"Camera {camera_name} failed to start")
            else:
                logging.debug(f"Camera {camera_name} disabled, skipping start_wait")

        # If any camera failed, raise an exception
        if failed_cameras:
            error_msg = f"Camera initialization failed for: {', '.join(failed_cameras)}"
            logging.error(error_msg)
            raise CameraError(error_msg)

    def stop_wait(self):
        if self.skip_camera:
            return

        logging.info("Waiting for camera processes to terminate...")
        cameras = [
            ("cam_high", self.camera_high),
            ("cam_left_wrist", self.camera_left_wrist),
            ("cam_right_wrist", self.camera_right_wrist)
        ]

        # First, try to signal all cameras to stop gracefully
        for camera_name, camera in cameras:
            if camera is not None and hasattr(camera, 'stop_event'):
                try:
                    logging.debug(f"Sending stop signal to camera {camera_name}")
                    camera.stop_event.set()
                except Exception as e:
                    logging.debug(f"Error signaling camera {camera_name}: {e}")

        # Give cameras a moment to respond to stop signals
        import time
        time.sleep(0.5)

        # Then wait for each camera to terminate
        for camera_name, camera in cameras:
            if camera is not None:
                try:
                    # First try gentle join
                    camera.join(timeout=2.0)
                    if camera.is_alive():
                        logging.warning(f"Camera {camera_name} not responding, terminating...")
                        camera.terminate()
                        camera.join(timeout=2.0)
                        if camera.is_alive():
                            logging.error(f"Camera {camera_name} still alive, force killing...")
                            camera.kill()
                            camera.join(timeout=1.0)  # Final wait after kill
                    else:
                        logging.debug(f"Camera {camera_name} terminated gracefully")
                except Exception as e:
                    logging.warning(f"Error during camera {camera_name} shutdown: {e}")
                    # Try force kill as last resort
                    try:
                        if camera.is_alive():
                            camera.kill()
                    except:
                        pass
            else:
                logging.debug(f"Camera {camera_name} disabled, skipping shutdown")

        logging.info("All camera processes terminated")

    def get_images(self):
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time

        Raises:
            CameraError: If any camera has encountered unrecoverable errors
        """
        if self.skip_camera:
            # Return dummy images when camera is disabled
            dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
            return {
                "cam_high": dummy_image,
                "cam_left_wrist": dummy_image,
                "cam_right_wrist": dummy_image,
            }

        image_dict = {}
        k = None
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)

        try:
            # Get images from enabled cameras only
            if self.camera_high is not None:
                self.last_camera_high_data = self.camera_high.get(k=k, out=self.last_camera_high_data)
                image_dict["cam_high"] = self.last_camera_high_data['left_color']
            else:
                image_dict["cam_high"] = dummy_image

            if self.camera_left_wrist is not None:
                self.last_camera_left_wrist_data = self.camera_left_wrist.get(k=k,out=self.last_camera_left_wrist_data)
                image_dict['cam_left_wrist'] = self.last_camera_left_wrist_data['left_color']
            else:
                image_dict['cam_left_wrist'] = dummy_image

            if self.camera_right_wrist is not None:
                self.last_camera_right_wrist_data = self.camera_right_wrist.get(k=k,out=self.last_camera_right_wrist_data)
                image_dict['cam_right_wrist'] = self.last_camera_right_wrist_data['left_color']
            else:
                image_dict['cam_right_wrist'] = dummy_image

        except CameraError as e:
            # Re-raise camera error to propagate to upper layers
            logging.error(f"Camera error in get_images: {e}")
            raise

        return image_dict

    def get_camera_error_status(self):
        """Get error status of all cameras.

        Returns:
            dict: Dictionary containing error status for each camera
        """
        if self.skip_camera:
            return {
                'cam_high': {'error_count': 0, 'last_error': None},
                'cam_left_wrist': {'error_count': 0, 'last_error': None},
                'cam_right_wrist': {'error_count': 0, 'last_error': None},
            }

        status = {}

        if self.camera_high is not None:
            status['cam_high'] = self.camera_high.get_error_status()
        else:
            status['cam_high'] = {'error_count': 0, 'last_error': 'Camera disabled'}

        if self.camera_left_wrist is not None:
            status['cam_left_wrist'] = self.camera_left_wrist.get_error_status()
        else:
            status['cam_left_wrist'] = {'error_count': 0, 'last_error': 'Camera disabled'}

        if self.camera_right_wrist is not None:
            status['cam_right_wrist'] = self.camera_right_wrist.get_error_status()
        else:
            status['cam_right_wrist'] = {'error_count': 0, 'last_error': 'Camera disabled'}

        return status

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
            print(f"{cam_name} {image_freq=:.2f}")
        print()
