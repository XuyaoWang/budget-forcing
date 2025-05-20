import asyncio
import atexit
import logging
import multiprocessing
import signal
import sys
import time
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_active_processes = []

def _cleanup_processes():
    """Terminate all active server processes"""
    for process in _active_processes:
        if process.is_alive():
            logger.info(f"Terminating server process (PID: {process.pid})")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"Force terminating server process (PID: {process.pid})")
                process.kill()

atexit.register(_cleanup_processes)

class BaseServer(ABC):
    """Abstract base class to manage a model serving instance in a separate process."""
    
    def __init__(self, model_path: str, port: int = 8000, host: str = '0.0.0.0', 
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, # Adjusted default based on common practice
                 max_seq_len: int = 32768,
                 dtype: str = 'auto', # More generic default
                 server_init_timeout: int = 300,
                 health_check_interval: int = 10,
                 health_check_timeout: float = 5.0,
                 health_check_path: str = "/health",
                 api_endpoint_path: str = "/v1/chat/completions",
                 model_name: Optional[str] = None,
                 framework_specific_args: Optional[Dict[str, Any]] = None,
                 **kwargs): # Catch any extra args, though framework_specific_args is preferred
        """
        Initialize a Base Server instance.
        
        Args:
            model_path: Path to the model to serve.
            port: Port to serve the API on.
            host: Host address to bind to (default 0.0.0.0 to accept from any IP).
            tensor_parallel_size: Number of GPUs for tensor parallelism (if applicable).
            pipeline_parallel_size: Number of GPUs for pipeline parallelism (if applicable).
            gpu_memory_utilization: Target fraction of GPU memory usage (if applicable).
            max_seq_len: Maximum sequence length supported by the model.
            dtype: Data type for model weights (e.g., 'auto', 'bfloat16', 'float16').
            server_init_timeout: Max seconds to wait for the server to become healthy.
            health_check_interval: Seconds between health checks during startup.
            health_check_timeout: Timeout in seconds for each individual health check request.
            health_check_path: URL path for health checks (relative to host:port).
            api_endpoint_path: URL path for the main API endpoint (relative to host:port).
            model_name: Optional name to identify the served model.
            framework_specific_args: Dictionary for framework-specific settings.
            **kwargs: Additional arguments, potentially passed to the underlying framework.
        """
        
        self.model_path = model_path
        self.port = port
        self.host = host
        self.model_name = model_name or model_path
        self.server_init_timeout = server_init_timeout
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        # Ensure health_check_path starts with / if not empty
        if health_check_path and not health_check_path.startswith('/'):
             health_check_path = '/' + health_check_path
        self.health_check_url = f"http://{host}:{port}{health_check_path}"
        # Ensure api_endpoint_path starts with / if not empty
        if api_endpoint_path and not api_endpoint_path.startswith('/'):
             api_endpoint_path = '/' + api_endpoint_path
        self.api_base = f"http://{host}:{port}{api_endpoint_path}"

        # Store common arguments that might be needed by subclasses
        self.common_args = {
            'model': model_path,
            'port': port,
            'host': host,
            'tensor_parallel_size': tensor_parallel_size,
            'pipeline_parallel_size': pipeline_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_seq_len': max_seq_len,
            'dtype': dtype,
            'served_model_name': [self.model_name] # Keep as list for potential multi-model serving
        }

        # Store framework specific args, merging kwargs for backward compatibility/flexibility
        self.framework_args = framework_specific_args or {}
        self.framework_args.update(kwargs) # Add any extra kwargs here

        self.process: Optional[multiprocessing.Process] = None
        self.mp_context = None
        self.started_event: Optional[multiprocessing.Event] = None
        self.exit_event: Optional[multiprocessing.Event] = None

    @abstractmethod
    def _build_server_args(self) -> Dict[str, Any]:
        """
        Prepare the dictionary of arguments needed by the specific server implementation's
        _start_server_instance and potentially _shutdown_server_instance methods.
        This method should combine self.common_args and self.framework_args as needed,
        performing any necessary transformations or validations specific to the framework.

        Returns:
            A dictionary of arguments for the server framework.
        """
        pass

    @staticmethod
    @abstractmethod
    async def _start_server_instance(server_args: Dict[str, Any]) -> Any:
        """
        Framework-specific method to start the actual server instance asynchronously.

        Args:
            server_args: Arguments returned by _build_server_args().

        Returns:
            An object representing the running server instance (e.g., a task, process handle)
            or None if not applicable. This object will be passed to _shutdown_server_instance.
        """
        pass

    @staticmethod
    @abstractmethod
    async def _shutdown_server_instance(server_instance: Any):
        """
        Framework-specific method to gracefully shut down the server instance asynchronously.

        Args:
            server_instance: The object returned by _start_server_instance.
        """
        pass

    @staticmethod
    def _run_server_process(cls, server_args: Dict[str, Any], started_event: multiprocessing.Event, exit_event: multiprocessing.Event):
        """
        The target function to run in the separate server process.
        This method orchestrates the server lifecycle using the abstract methods
        provided by the subclass `cls`.

        Args:
            cls: The subclass of BaseServer.
            server_args: Arguments including _base_config and framework args.
            started_event: Multiprocessing event to signal when the server is ready.
            exit_event: Multiprocessing event to signal the server process to shut down.
        """
        pid = os.getpid()
        _base_config = server_args.pop('_base_config', {})
        health_check_url = _base_config.get('health_check_url', 'http://localhost:8000/health')
        health_check_interval = _base_config.get('health_check_interval', 10)
        health_check_timeout = _base_config.get('health_check_timeout', 5.0)
        server_init_timeout = _base_config.get('server_init_timeout', 300)
        host = _base_config.get('host', 'unknown')
        port = _base_config.get('port', 'unknown')
        model_name = server_args.get('served_model_name', ['unknown'])[0]

        # --- Signal Handling ---
        def signal_handler(sig, frame):
            logger.info(f"Server subprocess {model_name} (PID: {pid}) received signal {sig}, setting exit event.")
            if not exit_event.is_set():
                 exit_event.set()
            # Allow the main loop/finally block handle the shutdown via _shutdown_server_instance

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"Server subprocess {model_name} (PID: {pid}) starting on {host}:{port}.")

        server_instance = None
        server_task = None
        health_check_task = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # --- Health Check Coroutine ---
            async def check_health():
                async with aiohttp.ClientSession() as session:
                    start_time = time.monotonic()
                    while time.monotonic() - start_time < server_init_timeout:
                        if exit_event.is_set():
                             logger.info(f"Exit requested during health check for {model_name} (PID: {pid}).")
                             return False
                        try:
                            # logger.debug(f"Checking health at {health_check_url}...")
                            async with session.get(health_check_url, timeout=health_check_timeout) as response:
                                if response.status == 200:
                                     logger.info(f"Server {model_name} at {host}:{port} (PID: {pid}) is healthy. Setting started_event.")
                                     if not started_event.is_set():
                                         started_event.set()
                                     return True # Healthy
                                else:
                                     logger.debug(f"Health check for {model_name} (PID: {pid}) failed status {response.status}. Retrying...")
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            logger.debug(f"Health check for {model_name} (PID: {pid}) failed: {e}. Retrying in {health_check_interval}s...")
                        except Exception as e:
                             logger.warning(f"Unexpected error during health check for {model_name} (PID: {pid}): {e}")
                             # Continue retrying unless it's a fatal error
                        
                        # Wait before next attempt, checking exit_event
                        wait_start_inner = time.monotonic()
                        while time.monotonic() - wait_start_inner < health_check_interval:
                            if exit_event.is_set():
                                logger.info(f"Exit requested during health check wait for {model_name} (PID: {pid}).")
                                return False
                            await asyncio.sleep(0.1)

                    logger.error(f"Server {model_name} at {host}:{port} (PID: {pid}) failed to become healthy within {server_init_timeout}s.")
                    return False # Unhealthy

            # --- Server Startup and Monitoring ---            
            async def main():
                nonlocal server_instance, server_task, health_check_task

                # Start the server instance using the subclass implementation
                # We run this first, the health check will verify it started correctly
                logger.info(f"Starting underlying server instance for {model_name} (PID: {pid})...")
                server_instance = await cls._start_server_instance(server_args)
                logger.info(f"Underlying server instance for {model_name} (PID: {pid}) started.")
                
                # Start health check
                health_check_task = asyncio.create_task(check_health())
                health_check_passed = await health_check_task

                if not health_check_passed:
                    logger.error(f"Health check failed or aborted for {model_name} (PID: {pid}). Signalling exit.")
                    if not exit_event.is_set():
                         exit_event.set() # Ensure shutdown sequence is triggered
                    # No need to sys.exit(1), finally block will handle cleanup
                    return # Exit the main coroutine

                # --- Main Loop (Keep process alive until exit signal) ---
                logger.info(f"Server process {model_name} (PID: {pid}) is running. Monitoring exit event.")
                while not exit_event.is_set():
                    # Optional: Add checks here if the server_instance itself can report errors
                    # or if it's a task that might complete unexpectedly.
                    await asyncio.sleep(0.5) # Prevent busy-waiting
                
                logger.info(f"Exit event set for {model_name} (PID: {pid}). Main loop ending.")

            loop.run_until_complete(main())

        except Exception as e:
            logger.error(f"Unhandled exception in server process {model_name} (PID: {pid}): {e}")
            logger.error(traceback.format_exc())
            if not exit_event.is_set():
                exit_event.set() # Ensure cleanup is triggered on unexpected errors
        finally:
            logger.info(f"Server process {model_name} (PID: {pid}) initiating shutdown sequence...")
            
            # Cancel health check if still running
            if health_check_task and not health_check_task.done():
                 logger.info(f"Cancelling health check task for {model_name} (PID: {pid}).")
                 health_check_task.cancel()
                 try:
                      # Use run_until_complete to wait for cancellation
                      loop.run_until_complete(asyncio.wait_for(health_check_task, timeout=5.0))
                 except (asyncio.CancelledError, asyncio.TimeoutError):
                      logger.warning(f"Health check task for {model_name} (PID: {pid}) did not cancel cleanly.")
                 except Exception as e:
                      logger.error(f"Error waiting for health check task cancellation for {model_name} (PID: {pid}): {e}")

            # Shutdown the server instance using the subclass implementation
            if server_instance is not None:
                logger.info(f"Shutting down underlying server instance for {model_name} (PID: {pid})...")
                try:
                    # Run the shutdown coroutine
                    shutdown_task = loop.create_task(cls._shutdown_server_instance(server_instance))
                    loop.run_until_complete(asyncio.wait_for(shutdown_task, timeout=15.0)) # 15s timeout for graceful shutdown
                    logger.info(f"Underlying server instance for {model_name} (PID: {pid}) shut down successfully.")
                except asyncio.TimeoutError:
                     logger.warning(f"Graceful shutdown timed out for {model_name} (PID: {pid}). Possible resource leak.")
                except Exception as e:
                    logger.error(f"Error shutting down server instance for {model_name} (PID: {pid}): {e}")
                    logger.error(traceback.format_exc())
            else:
                 logger.info(f"No server instance to shut down for {model_name} (PID: {pid}).")
            
            # Close the event loop
            logger.info(f"Closing asyncio event loop for {model_name} (PID: {pid}).")
            loop.close()
            logger.info(f"Server process {model_name} (PID: {pid}) exiting.")
            # Process exits naturally after this block

    async def start(self):
        """Start the server in a separate process and wait for it to become healthy."""
        if self.process is not None and self.process.is_alive():
            logger.warning(f"Server process for {self.model_name} is already running (PID: {self.process.pid})")
            return self.api_base
        
        # Create multiprocessing context, use spawn to avoid CUDA issues with some frameworks
        self.mp_context = multiprocessing.get_context('spawn')  
        
        # Create events using the same context
        self.started_event = self.mp_context.Event()
        self.exit_event = self.mp_context.Event()
        
        # Build arguments for the specific server implementation
        try:
            framework_args = self._build_server_args()
        except Exception as e:
            logger.error(f"Failed to build server arguments for {self.model_name}: {e}")
            raise ValueError(f"Error building server arguments: {e}") from e

        # Prepare args for the process runner
        server_process_args = framework_args.copy() # Avoid modifying the original dict
        server_process_args['_base_config'] = {
             'health_check_url': self.health_check_url,
             'server_init_timeout': self.server_init_timeout,
             'health_check_interval': self.health_check_interval,
             'health_check_timeout': self.health_check_timeout,
             'host': self.host, # Needed for logging/debugging in the subprocess
             'port': self.port, # Needed for logging/debugging in the subprocess
        }

        # Start server in a new process using the static method
        # Pass the class itself (self.__class__) to _run_server_process
        self.process = self.mp_context.Process(
            target=BaseServer._run_server_process, # Target the static method in BaseServer
            args=(self.__class__, server_process_args, self.started_event, self.exit_event),
            daemon=False # Allow server to create child processes if needed
        )
        logger.info(f"Starting server process for {self.model_name}...")
        self.process.start()
        
        global _active_processes
        _active_processes.append(self.process)
        
        # Wait for server process to signal readiness via the started_event
        logger.info(f"Waiting up to {self.server_init_timeout}s for server {self.model_name} (PID: {self.process.pid}) to signal readiness...")
        
        started = self.started_event.wait(timeout=self.server_init_timeout)
        
        # Post-start checks
        if not self.process.is_alive():
             exit_code = self.process.exitcode
             logger.error(f"Server process for {self.model_name} (PID: {self.process.pid}) died unexpectedly during startup (Exit code: {exit_code}). Check subprocess logs.")
             if self.process in _active_processes:
                 _active_processes.remove(self.process)
             self.process = None
             raise RuntimeError(f"Server process for {self.model_name} died unexpectedly during startup (Exit code: {exit_code}).")

        if not started:
            # Event timed out, but process is still alive - implies health check failed in subprocess
            # or subprocess never started correctly
            logger.error(f"Server {self.model_name} (PID: {self.process.pid}) failed to become healthy within {self.server_init_timeout} seconds (subprocess did not set started_event). Terminating process.")
            await self.stop() # Attempt graceful shutdown first
            if self.process and self.process.is_alive():
                 logger.warning(f"Force killing unresponsive server process {self.process.pid} for {self.model_name}.")
                 self.process.kill()
                 self.process.join(timeout=5)
            if self.process in _active_processes:
                 _active_processes.remove(self.process)
            self.process = None
            raise TimeoutError(f"Server {self.model_name} failed to start and become healthy within {self.server_init_timeout} seconds.")
        
        # Success Case: Process is alive and started_event is set
        logger.info(f"Server {self.model_name} (PID: {self.process.pid}) started successfully at {self.api_base}")
        return self.api_base
        
    async def stop(self):
        """Stop the running server process."""
        if self.process is None or not self.process.is_alive():
            logger.warning(f"No server process is running for {self.model_name} or it's already stopped.")
            self.process = None # Ensure state is clean
            return
            
        pid = self.process.pid
        logger.info(f"Stopping server {self.model_name} (PID: {pid})...")
        
        # Signal the process to exit cleanly via the exit_event
        if self.exit_event and not self.exit_event.is_set():
             logger.info(f"Setting exit event for server process {self.model_name} (PID: {pid}).")
             self.exit_event.set()
        elif not self.exit_event:
             logger.warning(f"Exit event not available for server {self.model_name} (PID: {pid}), attempting direct termination.")


        # Give the process some time to exit gracefully
        try:
            # Use process.join with a timeout. It waits for the process to terminate.
            self.process.join(timeout=20) # Wait up to 20 seconds for graceful shutdown (increased)
            if self.process.is_alive():
                 logger.warning(f"Server process {pid} for {self.model_name} did not exit gracefully after 20s, terminating...")
                 self.process.terminate() # Send SIGTERM
                 self.process.join(timeout=5) # Wait 5s for termination
                 if self.process.is_alive():
                      logger.warning(f"Server process {pid} for {self.model_name} did not terminate, killing...")
                      self.process.kill() # Send SIGKILL
                      self.process.join(timeout=5) # Wait 5s for kill
        except Exception as e:
             logger.error(f"Error occurred while stopping server process {pid} for {self.model_name}: {e}")
             # Ensure kill if it's somehow still alive after exception
             if self.process and self.process.is_alive():
                 logger.warning(f"Killing process {pid} due to error during stop sequence.")
                 self.process.kill()
                 self.process.join(timeout=5)

        if self.process and self.process.is_alive():
             logger.error(f"Failed to stop server process {pid} for {self.model_name} even after kill signal.")
        else:
            logger.info(f"Server {self.model_name} (PID: {pid}) stopped.")
        
        global _active_processes
        if self.process in _active_processes:
            try:
                _active_processes.remove(self.process)
            except ValueError: # Can happen if cleanup already ran
                pass
            
        self.process = None
        self.started_event = None
        self.exit_event = None


    def is_running(self) -> bool:
        """Check if the server process is currently running."""
        return self.process is not None and self.process.is_alive()

    def get_api_base(self) -> Optional[str]:
        """Get the API base URL if the server is configured."""
        return self.api_base

    def get_pid(self) -> Optional[int]:
        """Get the process ID of the server if it's running."""
        if self.is_running():
            return self.process.pid
        return None

    def __del__(self):
        """Ensure cleanup on object deletion, although atexit is preferred."""
        if self.process and self.process.is_alive():
             logger.warning(f"Server object for {self.model_name} deleted but process {self.process.pid} still running. Attempting cleanup via __del__.")
             # Avoid async call in __del__ if possible. Signal and terminate.
             if self.exit_event and not self.exit_event.is_set():
                  self.exit_event.set() # Try signalling first
                  time.sleep(0.1) # Brief pause
             self.process.terminate()
             self.process.join(timeout=5)
             if self.process.is_alive():
                  self.process.kill()