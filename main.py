import os
import asyncio
import argparse
import logging

import ray
from transformers import AutoTokenizer
from server.vllm_server import VLLMServer
from client.budget_forcing import BudgetForcingClient
import benchmarks 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Budget Forcing Evaluation Framework")
    
    # Server configuration
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model to serve")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Name to serve the model as (defaults to model path)")
    parser.add_argument("--port", type=int, default=8010,
                       help="Port to serve the API on")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host address to bind to")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                       help="Number of GPUs to use for pipeline parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                       help="Fraction of GPU memory to use")
    parser.add_argument("--limit-mm-per-prompt", type=str, default="image=10,video=10",
                       help="Limit on multimedia per prompt")
    parser.add_argument("--chat-template", type=str, default=None,
                       help="Path to chat template file")
    parser.add_argument("--max-seq-len", type=int, default=32768,
                       help="Maximum sequence length for the model")
    parser.add_argument("--enable-prefix-caching", action="store_true",
                       help="Whether to enable prefix caching")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       help="Data type for model weights")
    parser.add_argument("--disable-log-stats", action="store_true",
                       help="Disable log stats")
    parser.add_argument("--disable-log-requests", action="store_true",
                       help="Disable log requests")
    parser.add_argument("--disable-fastapi-docs", action="store_true",
                       help="Disable fastapi docs")
    parser.add_argument("--uvicorn-log-level", type=str, default="warning",
                       help="Uvicorn log level")
    parser.add_argument("--disable-frontend-multiprocessing", action="store_true",
                       help="Disable frontend multiprocessing to prevent nested multiprocessing")
    parser.add_argument("--server-init-timeout", type=int, default=300,
                       help="Timeout in seconds for server initialization (default: 300s = 5min)")
    
    # Server operation mode
    parser.add_argument("--no-server", action="store_true",
                       help="Don't start a server, use an existing one instead")
    parser.add_argument("--api-base", type=str, default=None,
                       help="Base URL for the API when using --no-server")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                       help="API key for the API")
    
    # Evaluation configuration
    parser.add_argument("--benchmark", type=str, required=True, choices=list(benchmarks.BenchmarkRegistry.list().keys()),
                       help="Benchmark to evaluate on")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to the benchmark dataset")
    parser.add_argument("--results-dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on")
    
    # Budget forcing parameters
    parser.add_argument("--num-ignore", type=int, default=0,
                       help="Number of times to ignore the stop token")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Temperature parameter for sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p parameter for sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                       help="Repetition penalty parameter")
    parser.add_argument("--max-tokens-thinking", type=int, default=32000,
                       help="Maximum tokens for thinking phase")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                       help="Path to the tokenizer (defaults to model path)")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                       help="Directory to cache results")
    parser.add_argument("--num-workers", type=int, default=10,
                       help="Maximum number of parallel workers")
    parser.add_argument("--reasoning", action="store_true", default=False,
                       help="Whether the model contains both bot and eot tokens")
    parser.add_argument("--bot", type=str, default="<think>",
                       help="Beginning of thinking token")
    parser.add_argument("--eot", type=str, default="</think>",
                       help="End of thinking token")
    
    # Visualization configuration
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Whether to generate visualization plots")
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Start the server if needed
    server = None
    api_base = args.api_base
    model_name = args.model_name if args.model_name else args.model_path
    
    if not args.no_server:
        logger.info(f"Starting vLLM server with model {args.model_path}")
        server = VLLMServer(
            model_path=args.model_path,
            model_name=model_name,
            port=args.port,
            host=args.host,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt=args.limit_mm_per_prompt,
            max_seq_len=args.max_seq_len,
            chat_template=args.chat_template,
            enable_prefix_caching=args.enable_prefix_caching,
            dtype=args.dtype,
            server_init_timeout=args.server_init_timeout
        )
        api_base = await server.start()
        logger.info(f"Server started at {api_base}")
    else:
        if not args.api_base:
            raise ValueError("--api-base must be provided when using --no-server")
        logger.info(f"Using existing server at {args.api_base}")
    
    try:
        client = BudgetForcingClient(
            api_base=api_base,
            api_key=args.api_key,
            model=model_name,
            tokenizer_path=args.tokenizer_path or args.model_path,
            cache_dir=f"{args.cache_dir}/{str(model_name).replace('/', '_')}",,
            num_workers=args.num_workers,
            reasoning=args.reasoning,
            bot=args.bot,
            eot=args.eot
        )
        
        # Create the evaluator
        evaluator_kwargs = {
            "model_name": model_name,
            "data_path": args.data_path,
            "results_dir": args.results_dir,
        }
        
        evaluator = benchmarks.BenchmarkRegistry.create(args.benchmark, **evaluator_kwargs)
        
        # Run the evaluation
        logger.info(f"Running evaluation on {args.benchmark} benchmark")
        
        # Load tokenizer for visualization only if needed
        tokenizer = None
        visualize = args.visualize and args.num_ignore > 0
        
        if visualize:
            try:
                tokenizer_path = args.tokenizer_path or args.model_path
                logger.info(f"Loading tokenizer from {tokenizer_path} for visualization")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for visualization: {str(e)}")
                visualize = False
        else:
            if not args.visualize:
                logger.info("Visualization is disabled")
            elif args.num_ignore <= 0:
                logger.info("Visualization requires num_ignore > 0, skipping")
                
        results = evaluator.run_evaluation(
            budget_forcing_client=client,
            num_ignore=args.num_ignore,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens_thinking=args.max_tokens_thinking,
            split=args.split,
            tokenizer=tokenizer,
            visualize=visualize
        )
        
        logger.info("Evaluation complete. Results:")
        for ignore_idx, result in results.items():
            logger.info(f"  num_ignore={ignore_idx}: accuracy={result.get('accuracy', 0):.4f}")
            
    finally:
        if server:
            logger.info("Stopping vLLM server")
            await server.stop()
            
        if ray.is_initialized():
            logger.info("Shutting down Ray")
            ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
