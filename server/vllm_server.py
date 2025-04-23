import asyncio
import atexit
import logging
import multiprocessing
import signal
import sys
import time
from typing import Any, Dict, Optional

import aiohttp

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

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
            logger.info(f"Terminating legacy vLLM server process (PID: {process.pid})")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"Force terminating vLLM server process (PID: {process.pid})")
                process.kill()

atexit.register(_cleanup_processes)

def run_vllm_server_process(args: Dict[str, Any], started_event: Any, exit_event: Any):
    """
    Function to run in a separate process to start the vLLM server.
    
    Args:
        args: Dictionary of arguments to pass to the vLLM server
        started_event: Event to signal when the server has started
        exit_event: Event to signal when the server should exit
    """

    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down vLLM server")
        exit_event.set()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    parsed_args = parser.parse_args([])
    
    for key, value in args.items():
        setattr(parsed_args, key, value)
    
    validate_parsed_serve_args(parsed_args)

    if not hasattr(parsed_args, 'host'):
        parsed_args.host = '0.0.0.0' 
    if not hasattr(parsed_args, 'port'):
        parsed_args.port = 8000
    
    # Run server in a separate task
    async def check_server_health():
        """Check if the server is healthy and set the started event when it is."""

        health_url = f"http://{parsed_args.host}:{parsed_args.port}/health"
            
        server_init_timeout = args.get('server_init_timeout', 300)
        health_check_interval = 10  # Check more frequently
        max_attempts = server_init_timeout // health_check_interval
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_attempts):
                if exit_event.is_set():
                    logger.info("Exit requested, aborting health check")
                    return
                
                try:
                    async with session.get(health_url, timeout=2.0) as response:
                        if response.status == 200:
                            logger.info(f"vLLM server is healthy at {health_url}")
                            started_event.set()
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    logger.info(f"Waiting for vLLM server to be ready. Current check time: {attempt * health_check_interval} seconds. Max timeout: {server_init_timeout} seconds")
                    await asyncio.sleep(health_check_interval)
            else:
                logger.error(f"vLLM server failed to become healthy within {server_init_timeout} seconds")
    
    async def run_server_with_health_check():
        """Run the server and health check together."""
        health_check_task = asyncio.create_task(check_server_health())
        
        try:
            server_task = asyncio.create_task(run_server(parsed_args))
            
            while not exit_event.is_set() and not server_task.done():
                await asyncio.sleep(1)
            
            if exit_event.is_set():
                logger.info("Exit requested, cancelling server task")
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            
            if not health_check_task.done():
                health_check_task.cancel()
                try:
                    await health_check_task
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            logger.error(f"Error running vLLM server: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # Run everything
    try:
        asyncio.run(run_server_with_health_check())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down vLLM server")
    finally:
        logger.info("vLLM server process exiting")


class VLLMServer:
    """Class to manage a vLLM server instance in a separate process."""
    
    def __init__(self, model_path: str, port: int = 8000, host: str = '0.0.0.0', 
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.7,
                 limit_mm_per_prompt: Optional[str] = None, 
                 chat_template: Optional[str] = None,
                 enable_prefix_caching: bool = True,
                 dtype: str = 'bfloat16',
                 disable_log_stats: bool = True,
                 disable_log_requests: bool = True,
                 disable_fastapi_docs: bool = True,
                 uvicorn_log_level: str = "warning",
                 server_init_timeout: int = 300,
                 model_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize a vLLM server with the given configuration.
        
        Args:
            model_path: Path to the model to serve
            port: Port to serve the API on
            host: Host address to bind to (default 0.0.0.0 to accept from any IP)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            limit_mm_per_prompt: Limit on multimedia per prompt (e.g., "image=10,video=10")
            chat_template: Path to chat template file
            enable_prefix_caching: Whether to enable prefix caching
            dtype: Data type for model weights
            disable_log_stats: Whether to disable log stats
            disable_log_requests: Whether to disable log requests
            disable_fastapi_docs: Whether to disable fastapi docs
            uvicorn_log_level: Uvicorn log level
            server_init_timeout: Timeout in seconds for server initialization (default: 300s = 5min)
            model_name: Name to serve the model as (defaults to model path)
            **kwargs: Additional arguments to pass to vLLM
        """
        
        self.args = {
            'model': model_path,
            'port': port,
            'host': host,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'enable_prefix_caching': enable_prefix_caching,
            'dtype': dtype,
            'disable_log_stats': disable_log_stats,
            'disable_log_requests': disable_log_requests,
            'disable_fastapi_docs': disable_fastapi_docs,
            'uvicorn_log_level': uvicorn_log_level,
            'server_init_timeout': server_init_timeout,
        }
        
        if model_name:
            self.args['served_model_name'] = [model_name]
        else:
            self.args['served_model_name'] = [model_path]
                
            
        if limit_mm_per_prompt:
            limit_dict = {}
            for item in limit_mm_per_prompt.split(','):
                key, value = item.split('=')
                limit_dict[key] = int(value)
            self.args['limit_mm_per_prompt'] = limit_dict
        else:
            self.args['limit_mm_per_prompt'] = {}
            
        if chat_template:
            self.args['chat_template'] = chat_template
            
        self.args.update(kwargs)
        
        self.api_base = f"http://{host}:{port}/v1/chat/completions"
        
        self.process = None
        self.mp_context = None
        self.started_event = None
        self.exit_event = None
        
    async def start(self):
        """Start the vLLM server in a separate process."""
        if self.process is not None and self.process.is_alive():
            logger.warning("vLLM server is already running")
            return self.api_base
        
        # Create multiprocessing context, use spawn to avoid CUDA issues
        self.mp_context = multiprocessing.get_context('spawn')  
        
        # Create events using the same context
        self.started_event = self.mp_context.Event()
        self.exit_event = self.mp_context.Event()
        
        # Start server in a new process
        self.process = self.mp_context.Process(
            target=run_vllm_server_process,
            args=(self.args, self.started_event, self.exit_event),
            daemon=False  # vLLM needs to create child processes, so we can't use daemon processes
        )
        self.process.start()
        
        global _active_processes
        _active_processes.append(self.process)
        
        # Wait for server to become ready
        server_init_timeout = self.args.get('server_init_timeout', 300)
        logger.info("Waiting for vLLM server process to signal readiness...")
        
        started = self.started_event.wait(timeout=server_init_timeout)
        
        if not started:
            # Event timed out
            if self.process.is_alive():
                logger.error(f"vLLM server failed to start within {server_init_timeout} seconds (timeout waiting for event). Terminating process.")
                self.exit_event.set() # Signal the process to exit if possible
                self.process.terminate()
                self.process.join(timeout=5) # Wait briefly for termination
                if self.process.is_alive():
                    self.process.kill() # Force kill if terminate fails
            else:
                 logger.error(f"vLLM server failed to start within {server_init_timeout} seconds and process died.")

            if self.process in _active_processes:
                _active_processes.remove(self.process)
            raise TimeoutError(f"vLLM server failed to start within {server_init_timeout} seconds")
        
        if not self.process.is_alive():
            logger.error("vLLM server process died unexpectedly after signaling start.")
            if self.process in _active_processes:
                _active_processes.remove(self.process)
            raise RuntimeError("vLLM server process died unexpectedly")
        
        logger.info(f"vLLM server started successfully at {self.api_base}")
        return self.api_base
        
    async def stop(self):
        """Stop the vLLM server."""
        if self.process is None or not self.process.is_alive():
            logger.warning("No vLLM server process is running")
            return
            
        # Signal the process to exit cleanly
        logger.info("Stopping vLLM server...")
        self.exit_event.set()
        
        # Give the process some time to exit gracefully
        for _ in range(10):  # Wait up to 10 seconds
            if not self.process.is_alive():
                break
            await asyncio.sleep(1)
        
        if self.process.is_alive():
            logger.warning("vLLM server did not exit gracefully, terminating...")
            self.process.terminate()
            
            await asyncio.sleep(5)
            if self.process.is_alive():
                logger.warning("vLLM server did not terminate, killing...")
                self.process.kill()
        
        self.process.join(timeout=5)
        logger.info("vLLM server stopped")
        
        global _active_processes
        if self.process in _active_processes:
            _active_processes.remove(self.process)
            
        self.process = None


async def start_server_main():
    """Example usage of the VLLMServer class."""
    # Example configuration
    server = VLLMServer(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        model_name="qwen-vl-3b",
        port=8000,
        host="0.0.0.0",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.7,
        limit_mm_per_prompt="image=10,video=10",
        chat_template=None,
        enable_prefix_caching=True,
        dtype="bfloat16",
        disable_log_stats=True,
        disable_log_requests=True,
        disable_fastapi_docs=True,
        uvicorn_log_level="warning",
        server_init_timeout=300
    )
    
    try:
        api_base = await server.start()
        logger.info(f"Server started at {api_base}")
        
        # Keep server running for a while
        await asyncio.sleep(60)
    finally:
        await server.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(start_server_main())