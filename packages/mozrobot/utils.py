# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import time
import logging
import logging.handlers
import os
import subprocess
import ctypes
from typing import List
try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import threading

# Linux nanosleep 实现
_libc = ctypes.CDLL("libc.so.6")

class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

_nanosleep = _libc.nanosleep
_nanosleep.argtypes = [ctypes.POINTER(timespec), ctypes.POINTER(timespec)]
_nanosleep.restype = ctypes.c_int

# Global flag to track if memory has been locked in this process
_memory_locked = False

def lock_memory():
    global _memory_locked

    if _memory_locked:
        logging.debug("Memory already locked in this process, skipping")
        return

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        MCL_CURRENT = 1
        MCL_FUTURE = 2

        # lock current and future memory pages
        result = libc.mlockall(MCL_CURRENT | MCL_FUTURE)

        if result == 0:
            _memory_locked = True
            logging.info("Lock memory success")
        else:
            logging.warning(f"Lock memory failed: {result}")

    except Exception as e:
        logging.warning(f"Lock memory failed: {e}")

def unlock_memory():
    global _memory_locked

    if not _memory_locked:
        logging.debug("Memory not locked in this process, skipping unlock")
        return

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.munlockall()
        _memory_locked = False
        logging.info("Unlock memory success")
    except Exception as e:
        logging.warning(f"Unlock memory failed: {e}")

def realtime_precise_sleep(duration_seconds):
    """
    Linux专用高精度sleep，直接使用nanosleep系统调用
    """
    if duration_seconds <= 0:
        return

    # 分解为秒和纳秒
    seconds = int(duration_seconds)
    nanoseconds = int((duration_seconds - seconds) * 1_000_000_000)

    req = timespec(seconds, nanoseconds)
    rem = timespec(0, 0)

    # 调用nanosleep，自动处理信号中断
    while True:
        result = _nanosleep(ctypes.byref(req), ctypes.byref(rem))
        if result == 0:
            break  # 成功完成
        # 如果被信号中断(EINTR)，继续等待剩余时间
        req = rem
        if req.tv_sec == 0 and req.tv_nsec == 0:
            break

def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def safe_disconnect(func):
    # TODO(aliberts): Allow to pass custom exceptions
    # (e.g. ThreadServiceExit, KeyboardInterrupt, SystemExit, UnpluggedError, DynamixelCommError)
    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if robot.is_connected:
                robot.disconnect()
            raise e

    return wrapper


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


class TimingProfiler:
    """A profiler class for timing different operations in the control loop."""

    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.timing_counts = {}  # Track how many times each timing was recorded
        self.timing_history = {}  # Store historical data for each operation

    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._record_timing(operation_name, duration)

    def start(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.perf_counter()

    def stop(self, operation_name: str):
        """Stop timing an operation and record the duration."""
        if operation_name in self.start_times:
            duration = time.perf_counter() - self.start_times[operation_name]
            self._record_timing(operation_name, duration)
            del self.start_times[operation_name]
        else:
            logging.warning(
                f"Attempted to stop timing for '{operation_name}' which was not started"
            )

    def _record_timing(self, operation_name: str, duration: float):
        """Internal method to record timing data with history support."""
        # Initialize history list if not exists
        if operation_name not in self.timing_history:
            self.timing_history[operation_name] = []

        # Add to history
        self.timing_history[operation_name].append(duration)

        # Update current timing (latest value)
        self.timings[operation_name] = duration

        # Update count
        self.timing_counts[operation_name] = len(self.timing_history[operation_name])

    # Legacy methods for backward compatibility
    def start_timing(self, operation_name: str):
        """Start timing an operation manually."""
        self.start(operation_name)

    def end_timing(self, operation_name: str):
        """End timing an operation manually."""
        self.stop(operation_name)

    def get_timing(self, operation_name: str) -> float:
        """Get the latest timing for a specific operation."""
        if operation_name in self.timing_history and self.timing_history[operation_name]:
            return self.timing_history[operation_name][-1]  # Return the last (latest) entry
        return 0.0

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.start_times.clear()
        self.timing_counts.clear()
        self.timing_history.clear()

    def get_timing_summary(self) -> dict:
        """Get a summary of all timings (latest values)."""
        summary = {}
        for operation_name in self.timing_history:
            if self.timing_history[operation_name]:
                summary[operation_name] = self.timing_history[operation_name][-1]
        return summary

    def format_timing_info(
        self, dt_s: float, fps: float, drift_threshold: float = 0.05, show_counts: bool = False
    ) -> str:
        """Format timing information for logging."""
        actual_fps = 1 / dt_s if dt_s > 0 else 0
        is_drifting = actual_fps < fps - fps * drift_threshold

        info_parts = [f"fps: {actual_fps:3.1f}hz (dt: {dt_s * 1000:5.2f}ms, {actual_fps - fps:.1f}hz)"]

        # Add detailed timing breakdown if we have timing data
        timing_details = []
        total_measured_time = 0
        for operation_name in self.timing_history:
            if self.timing_history[operation_name]:
                duration = self.timing_history[operation_name][-1]  # Get latest timing
                if duration > 0:
                    if show_counts and operation_name in self.timing_counts:
                        count = self.timing_counts[operation_name]
                        if count > 1:
                            avg_duration = self.get_timing_average(operation_name)
                            timing_details.append(f"{operation_name}: {duration * 1000:4.2f}ms({count}x, avg:{avg_duration * 1000:4.2f}ms)")
                        else:
                            timing_details.append(f"{operation_name}: {duration * 1000:4.2f}ms")
                    else:
                        timing_details.append(f"{operation_name}: {duration * 1000:4.2f}ms")
                    total_measured_time += duration

        if timing_details:
            info_parts.append(" | ".join(timing_details))

            # Add unmeasured time if significant
            unmeasured_time = dt_s - total_measured_time
            if unmeasured_time > 0.001:  # Only show if > 1ms
                info_parts.append(f"other: {unmeasured_time * 1000:4.2f}ms")

        info_str = " | ".join(info_parts)
        return colored(info_str, "yellow") if is_drifting else info_str

    def add_timing(self, operation_name: str, duration: float):
        """Add a timing measurement (supports multiple entries for same operation)."""
        self._record_timing(operation_name, duration)

    def add_to_timing(self, operation_name: str, duration: float):
        """Add a duration value to an existing timing key (accumulates values)."""
        self._record_timing(operation_name, duration)

    def get_timing_average(self, operation_name: str) -> float:
        """Get the average timing for an operation from its history."""
        if operation_name in self.timing_history and self.timing_history[operation_name]:
            history = self.timing_history[operation_name]
            return sum(history) / len(history)
        return 0.0

    def get_timing_max(self, operation_name: str) -> float:
        """Get the maximum timing for an operation from its history."""
        if operation_name in self.timing_history and self.timing_history[operation_name]:
            return max(self.timing_history[operation_name])
        return 0.0

    def get_timing_min(self, operation_name: str) -> float:
        """Get the minimum timing for an operation from its history."""
        if operation_name in self.timing_history and self.timing_history[operation_name]:
            return min(self.timing_history[operation_name])
        return 0.0

    def get_timing_count(self, operation_name: str) -> int:
        """Get the number of times a timing was recorded for an operation."""
        return self.timing_counts.get(operation_name, 0)

    def get_timing_history(self, operation_name: str) -> list:
        """Get the complete timing history for an operation."""
        return self.timing_history.get(operation_name, []).copy()

    def clear_timing_history(self, operation_name: str):
        """Clear the timing history for a specific operation."""
        if operation_name in self.timing_history:
            self.timing_history[operation_name].clear()
            self.timing_counts[operation_name] = 0
            if operation_name in self.timings:
                del self.timings[operation_name]

def log_drift_dt_info(dt_s, fps, id=None, drift_threshold=0.05, profiler: TimingProfiler = None):
    """Log drift control information with optional detailed timing breakdown."""
    actual_fps = 1.0 / dt_s
    if abs(actual_fps - fps) <= fps * drift_threshold:
        return

    info_str = f"[{id}] " if id is not None else ""
    info_str += "drift "

    if profiler is not None:
        info_str += profiler.format_timing_info(dt_s, fps, drift_threshold)
    else:
        info_str += f"fps: {actual_fps:3.1f}hz (dt: {dt_s * 1000:5.2f}ms, {actual_fps - fps:.1f}hz)"
        info_str = colored(info_str, "yellow")

    logging.info(info_str)

# To enable soft real-time, must run tools/setup_rtprio.sh firstly
def set_soft_realtime(priority: int = 20, thread_id: str = "thread"):
    import threading
    tid = threading.get_native_id()
    try:
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(priority))
        logging.info(f"Soft real-time enabled for {thread_id} with priority {priority} for pid {tid}")
    except Exception as e:
        logging.error(f"Failed to set soft real-time: {e} for {thread_id} with priority {priority} for pid {tid}")

    lock_memory()

def pre_initialize_queue_feeder_threads(queues: list, thread_id: str = "thread"):
    """
    Pre-initialize QueueFeederThread by triggering their creation before setting real-time scheduling.
    This ensures QueueFeederThread will be created with normal (SCHED_OTHER) scheduling.

    Args:
        queues: List of multiprocessing.Queue objects to initialize
        thread_id: Identifier for logging purposes
    """
    try:
        import threading

        # Get initial thread count
        initial_thread_count = threading.active_count()
        initial_threads = {t.name for t in threading.enumerate()}

        logging.info(f"[{thread_id}] Pre-initializing queue feeder threads...")
        logging.info(f"[{thread_id}] Initial thread count: {initial_thread_count}")
        logging.info(f"[{thread_id}] Initial threads: {initial_threads}")

                # Trigger QueueFeederThread creation by putting dummy data into each queue
        dummy_data = "__QUEUE_INIT_DUMMY__"

        for i, queue in enumerate(queues):
            try:
                # Put dummy data to trigger QueueFeederThread creation
                queue.put(dummy_data, block=False)

                # Small delay to ensure thread creation
                time.sleep(0.002)

                # Remove the dummy data immediately to avoid contaminating the queue
                try:
                    retrieved_data = queue.get(block=False)
                    if retrieved_data == dummy_data:
                        logging.info(f"[{thread_id}] Queue {i} successfully pre-initialized and cleaned")
                    else:
                        # Put back the non-dummy data
                        queue.put(retrieved_data, block=False)
                        logging.warning(f"[{thread_id}] Queue {i} had existing data, putting it back")
                except:
                    # This is actually expected - the dummy data might be consumed by another process
                    logging.info(f"[{thread_id}] Queue {i} pre-initialized (dummy data consumed by other process)")

            except Exception as e:
                logging.warning(f"[{thread_id}] Failed to pre-initialize queue {i}: {e}")

        # Wait a bit for threads to be fully created
        time.sleep(0.01)

        # Check final thread count
        final_thread_count = threading.active_count()
        final_threads = {t.name for t in threading.enumerate()}
        new_threads = final_threads - initial_threads

        logging.info(f"[{thread_id}] Final thread count: {final_thread_count}")
        logging.info(f"[{thread_id}] New threads created: {new_threads}")

        if new_threads:
            logging.info(f"[{thread_id}] Successfully pre-initialized {len(new_threads)} queue feeder threads")
        else:
            logging.info(f"[{thread_id}] No new threads created (may already exist)")

        return True

    except Exception as e:
        logging.error(f"[{thread_id}] Failed to pre-initialize queue feeder threads: {e}")
        return False

def set_soft_realtime_with_queue_protection(queues: list, priority: int = 20, thread_id: str = "thread"):
    """
    Set soft real-time scheduling after pre-initializing queue feeder threads.
    This ensures QueueFeederThread will not be affected by real-time scheduling.

    Args:
        queues: List of multiprocessing.Queue objects to protect
        priority: Real-time priority to set
        thread_id: Identifier for logging purposes
    """
    try:
        # Step 1: Pre-initialize queue feeder threads with normal scheduling
        success = pre_initialize_queue_feeder_threads(queues, thread_id)
        if not success:
            logging.warning(f"[{thread_id}] Queue pre-initialization failed, proceeding with caution")

        # Step 2: Set real-time scheduling for current thread
        set_soft_realtime(priority, thread_id)

        logging.info(f"[{thread_id}] Soft real-time with queue protection completed")
        return True

    except Exception as e:
        logging.error(f"[{thread_id}] Failed to set soft real-time with queue protection: {e}")
        return False

def set_cpu_affinity(cpu_idxs: List[int], thread_id: str = "thread"):
    if not hasattr(set_cpu_affinity, "available_core_ids"):
        # Get available CPU cores instead of assuming sequential numbering
        try:
            # Use lscpu to get the list of online CPUs
            result = subprocess.run(["lscpu", "-p=CPU"], capture_output=True, text=True, check=True)
            # Parse the output to get available CPU IDs (skip comment lines starting with #)
            set_cpu_affinity.available_core_ids = [int(line) for line in result.stdout.splitlines() if not line.startswith('#')]
        except (subprocess.SubprocessError, ValueError):
            logging.warning(f"Failed to get physical CPU cores, falling back to logical cores")
            import multiprocessing
            set_cpu_affinity.available_core_ids = list(range(multiprocessing.cpu_count()))
        logging.info(f"Available CPU cores: {set_cpu_affinity.available_core_ids}")

    # Validate and map the requested CPU indices to actual available cores
    actual_cpu_idxs = []
    for idx in cpu_idxs:
        if idx < 0 or idx >= len(set_cpu_affinity.available_core_ids):
            logging.warning(f"CPU index {idx} is out of range (0-{len(set_cpu_affinity.available_core_ids)-1}), skipping")
            continue
        actual_cpu_idxs.append(set_cpu_affinity.available_core_ids[idx])

    if not actual_cpu_idxs:
        logging.warning(f"No valid CPU indices provided, using all available cores")
        actual_cpu_idxs = set_cpu_affinity.available_core_ids

    try:
        os.sched_setaffinity(0, actual_cpu_idxs)
        logging.info(
            f"CPU affinity set for {thread_id} with CPU indices {actual_cpu_idxs}"
        )
    except Exception as e:
        logging.error(
            f"Failed to set CPU affinity: {e} for {thread_id} with CPU indices {actual_cpu_idxs}"
        )

def get_low_priority_task_bind_cpu_idxs_list():
    # we use the last 4 cores for low priority task bind
    if not hasattr(get_low_priority_task_bind_cpu_idxs_list, "cpu_idxs"):
        # Get available CPU cores instead of assuming sequential numbering
        try:
            import subprocess
            # Use lscpu to get the list of online CPUs
            result = subprocess.run(["lscpu", "-p=CPU"], capture_output=True, text=True, check=True)
            # Parse the output to get available CPU IDs (skip comment lines starting with #)
            available_cores = [int(line) for line in result.stdout.splitlines() if not line.startswith('#')]
            # Take the last 4 available cores or all if less than 4
            get_low_priority_task_bind_cpu_idxs_list.cpu_idxs = available_cores[-4:]
        except (subprocess.SubprocessError, ValueError):
            import multiprocessing
            # Fallback to simple counting if lscpu fails
            total_cores = multiprocessing.cpu_count()
            get_low_priority_task_bind_cpu_idxs_list.cpu_idxs = range(max(0, total_cores - 4), total_cores)
        except Exception as e:
            logging.info(f"Failed to get low priority task bind CPU indices: {e}")

    # Use cached value if available
    return get_low_priority_task_bind_cpu_idxs_list.cpu_idxs

def setup_low_priority_task(decrease_value: int = 10, thread_id: str = "thread"):
    try:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        os.nice(decrease_value)
    except Exception as e:
        logging.error(
            f"Failed to set nice value: {e} for {thread_id} with priority {decrease_value}"
        )

    logging.info(f"Low priority task setup completed for {thread_id} with decrease value {decrease_value}")

    bind_cpu_idxs = get_low_priority_task_bind_cpu_idxs_list()
    set_cpu_affinity(bind_cpu_idxs, thread_id)

    logging.info(f"Low priority task bind CPU indices set for {thread_id} with CPU indices {bind_cpu_idxs}")


def create_multiprocess_log_handler():
    """Create a multiprocess-safe log handler that truly clones the caller's existing handlers."""
    import multiprocessing
    import threading
    import gc

    # Create a shared queue for multiprocess logging
    log_queue = multiprocessing.Queue()

    # Get the current root logger
    root_logger = logging.getLogger()
    
    def _extract_active_queue_listeners():
        """Extract active QueueListener instances to access their handlers."""
        listeners = []
        
        import warnings
        warnings.filterwarnings("ignore", message=".*torch.distributed.reduce_op.*")
        # Search through all objects in garbage collector for active QueueListener instances
        for obj in gc.get_objects():
            if (isinstance(obj, logging.handlers.QueueListener) and 
                hasattr(obj, 'handlers') and 
                hasattr(obj, '_thread') and 
                obj._thread and obj._thread.is_alive()):
                listeners.append(obj)
        
        return listeners
    
    def _clone_handler_with_multiprocess_formatter(original_handler):
        """Clone a handler and adapt its formatter for multiprocess use."""
        
        # Extract the original formatter's format function if it exists
        original_formatter = original_handler.formatter
        original_format_func = None
        
        if original_formatter and hasattr(original_formatter, 'format') and callable(original_formatter.format):
            original_format_func = original_formatter.format
        
        def _create_enhanced_formatter(original_func):
            def enhanced_format(record):
                # Ensure process_id and native_tid are available
                if not hasattr(record, 'process_id'):
                    record.process_id = os.getpid()
                if not hasattr(record, 'native_tid'):
                    record.native_tid = threading.get_native_id()
                
                if original_func:
                    return original_func(record)
                else:
                    # Fallback formatter
                    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    fnameline = f"{record.pathname}:{record.lineno}"
                    level_short = record.levelname[0]
                    return f"{level_short} {record.process_id}:{record.native_tid} {dt} {fnameline[-15:]:>15} {record.msg}"
            
            formatter = logging.Formatter()
            formatter.format = enhanced_format
            return formatter

        # Clone based on handler type
        new_handler = None
        
        try:
            if isinstance(original_handler, logging.StreamHandler) and not isinstance(original_handler, logging.FileHandler):
                new_handler = logging.StreamHandler(original_handler.stream)
            elif isinstance(original_handler, logging.handlers.TimedRotatingFileHandler):
                new_handler = logging.handlers.TimedRotatingFileHandler(
                    filename=original_handler.baseFilename,
                    when=original_handler.when,
                    interval=original_handler.interval,
                    backupCount=original_handler.backupCount,
                    encoding=getattr(original_handler, 'encoding', 'utf-8')
                )
                if hasattr(original_handler, 'suffix'):
                    new_handler.suffix = original_handler.suffix
            elif isinstance(original_handler, logging.handlers.RotatingFileHandler):
                new_handler = logging.handlers.RotatingFileHandler(
                    filename=original_handler.baseFilename,
                    maxBytes=original_handler.maxBytes,
                    backupCount=original_handler.backupCount,
                    encoding=getattr(original_handler, 'encoding', 'utf-8')
                )
            elif isinstance(original_handler, logging.FileHandler):
                new_handler = logging.FileHandler(
                    filename=original_handler.baseFilename,
                    encoding=getattr(original_handler, 'encoding', 'utf-8')
                )
            
            if new_handler:
                new_handler.setLevel(original_handler.level)
                new_handler.setFormatter(_create_enhanced_formatter(original_format_func))
                return new_handler
                
        except Exception as e:
            logging.warning(f"Failed to clone handler {type(original_handler).__name__}: {e}")
        
        return None

    # Try to extract real handlers from active QueueListeners
    target_handlers = []
    active_listeners = _extract_active_queue_listeners()

    logging.debug(f"DEBUG: Found {len(active_listeners)} active QueueListener instances")

    for listener in active_listeners:
        logging.debug(f"DEBUG: QueueListener has {len(listener.handlers)} handlers")
        for i, handler in enumerate(listener.handlers):
            logging.debug(f"DEBUG: Handler {i}: {type(handler).__name__}")
            if hasattr(handler, 'baseFilename'):
                logging.debug(f"DEBUG: Handler {i} file: {handler.baseFilename}")

            cloned_handler = _clone_handler_with_multiprocess_formatter(handler)
            if cloned_handler:
                target_handlers.append(cloned_handler)
                logging.debug(f"Successfully cloned {type(handler).__name__} -> {type(cloned_handler).__name__}")

    # If no handlers were found through QueueListener extraction, fall back to basic handlers
    if not target_handlers:
        logging.debug("No handlers extracted from QueueListeners, creating fallback handlers")

        # Create basic console handler
        console_handler = logging.StreamHandler()

        def fallback_format(record):
            if not hasattr(record, 'process_id'):
                record.process_id = os.getpid()
            if not hasattr(record, 'native_tid'):
                record.native_tid = threading.get_native_id()

            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            fnameline = f"{record.pathname}:{record.lineno}"
            level_short = record.levelname[0]
            return f"{level_short} {record.process_id}:{record.native_tid} {dt} {fnameline[-15:]:>15} {record.msg}"

        formatter = logging.Formatter()
        formatter.format = fallback_format
        console_handler.setFormatter(formatter)
        console_handler.setLevel(root_logger.level)
        target_handlers.append(console_handler)

    logging.debug(f"Multiprocess logging: Created {len(target_handlers)} handlers")
    for handler in target_handlers:
        logging.debug(f"  - {type(handler).__name__}: Level {handler.level}")
        if hasattr(handler, 'baseFilename'):
            logging.debug(f"    File: {handler.baseFilename}")

    # Create queue listener that will handle logs from child processes
    queue_listener = logging.handlers.QueueListener(
        log_queue,
        *target_handlers,
        respect_handler_level=True
    )
    queue_listener.start()

    return log_queue, queue_listener

def setup_child_process_logging(log_queue):
    """Setup logging for child process using the shared log queue."""
    # Add custom LogRecord factory to include native thread ID and process ID
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.native_tid = threading.get_native_id()
        record.process_id = os.getpid()
        return record
    logging.setLogRecordFactory(record_factory)

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup queue handler to send logs to main process
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logging.getLogger().addHandler(queue_handler)
    logging.getLogger().setLevel(logging.INFO)
