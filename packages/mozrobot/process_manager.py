#!/usr/bin/env python3
"""
Process Manager: For handling graceful shutdown of multiprocess applications

This module provides tools to properly handle KeyboardInterrupt and other signals,
ensuring child processes can shut down gracefully without generating stack traces.
"""

import signal
import logging
import atexit
import multiprocessing as mp
from typing import List, Optional
import time

logger = logging.getLogger(__name__)

class ProcessManager:
    """Manages the lifecycle of child processes, ensuring graceful shutdown"""

    def __init__(self):
        self._processes: List[mp.Process] = []
        self._queues: List[mp.Queue] = []
        self._shutdown_requested = False
        self._original_sigint_handler = None

        # Register signal handlers and exit cleanup
        self._setup_signal_handlers()
        atexit.register(self._cleanup_on_exit)

    def _setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(signum, frame):
            logger.info("🛑 Received interrupt signal, starting graceful shutdown...")
            self._shutdown_requested = True
            self.shutdown_all()

        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    def register_process(self, process: mp.Process):
        """Register a process that needs to be managed"""
        self._processes.append(process)
        logger.debug(f"Registered process: {process.name} (PID: {process.pid})")

    def register_queue(self, queue: mp.Queue):
        """Register a queue that needs to be managed"""
        self._queues.append(queue)

    def send_shutdown_signals(self):
        """Send shutdown signals to all queues"""
        for queue in self._queues:
            try:
                queue.put([("shutdown", 0)], timeout=0.1)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal: {e}")

    def shutdown_all(self, timeout: float = 5.0):
        """Shutdown all managed processes"""
        if not self._processes and not self._queues:
            return

        logger.info("Starting shutdown of all child processes...")

        # Step 1: Send graceful shutdown signals
        self.send_shutdown_signals()

        # Step 2: Wait for processes to exit naturally
        start_time = time.time()
        remaining_processes = [p for p in self._processes if p.is_alive()]

        while remaining_processes and (time.time() - start_time) < timeout/2:
            time.sleep(0.1)
            remaining_processes = [p for p in remaining_processes if p.is_alive()]

        if not remaining_processes:
            logger.debug("All processes shut down gracefully")
            return

        # Step 3: Terminate remaining processes
        logger.debug(f"Terminating {len(remaining_processes)} remaining processes...")
        for process in remaining_processes:
            try:
                if process.is_alive():
                    logger.debug(f"Terminating process: {process.name} (PID: {process.pid})")
                    process.terminate()
            except Exception as e:
                logger.warning(f"Failed to terminate process: {e}")

        # Step 4: Wait for termination to complete
        start_time = time.time()
        while remaining_processes and (time.time() - start_time) < timeout/2:
            time.sleep(0.1)
            remaining_processes = [p for p in remaining_processes if p.is_alive()]

        # Step 5: Force kill processes that are still running
        if remaining_processes:
            logger.warning(f"Force killing {len(remaining_processes)} stubborn processes...")
            for process in remaining_processes:
                try:
                    if process.is_alive():
                        logger.debug(f"Killing process: {process.name} (PID: {process.pid})")
                        process.kill()
                        process.join(timeout=1.0)
                except Exception as e:
                    logger.warning(f"Failed to kill process: {e}")

        logger.info("Process shutdown completed")

    def _cleanup_on_exit(self):
        """Cleanup function on exit"""
        if self._processes:
            logger.debug("Cleaning up remaining processes on exit...")
            self.shutdown_all(timeout=3.0)

    def restore_signal_handlers(self):
        """Restore original signal handlers"""
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)


# Global process manager instance
_global_process_manager = None

def get_process_manager() -> ProcessManager:
    """Get the global process manager instance"""
    global _global_process_manager
    if _global_process_manager is None:
        _global_process_manager = ProcessManager()
    return _global_process_manager

def register_process(process: mp.Process):
    """Register a process with the global process manager"""
    get_process_manager().register_process(process)

def register_queue(queue: mp.Queue):
    """Register a queue with the global process manager"""
    get_process_manager().register_queue(queue)

def shutdown_all_processes():
    """Shutdown all registered processes"""
    get_process_manager().shutdown_all()