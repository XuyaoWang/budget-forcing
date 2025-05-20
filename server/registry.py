import logging
from typing import Dict, Type, Any, Optional

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServerRegistry:
    """Registry for server implementations."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str = None):
        """
        Decorator to register a server implementation class.

        Args:
            name: Name to register the server under (defaults to class name)

        Returns:
            Decorator function
        """
        def decorator(server_class):
            registered_name = name or server_class.__name__
            cls._registry[registered_name] = server_class
            logger.debug(f"Registered server implementation: {registered_name}")
            return server_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        """
        Get a server implementation class by name.

        Args:
            name: Name of the server implementation

        Returns:
            Server implementation class

        Raises:
            KeyError: If the server implementation is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Server implementation '{name}' not registered. Available servers: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> Dict[str, Type]:
        """
        List all registered server implementations.

        Returns:
            Dictionary mapping server names to implementation classes
        """
        return cls._registry.copy()

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Create an instance of a server implementation.

        Args:
            name: Name of the server implementation
            **kwargs: Arguments to pass to the server constructor

        Returns:
            Instance of the server implementation
        """
        server_class = cls.get(name)
        return server_class(**kwargs)


# Import all server implementations here to register them
from server.vllm.vllm_server import VLLMServer
# from server.lmdeploy.lmdeploy_server import LMDeployServer

# If VLLMServer uses @ServerRegistry.register('vllm'), this is redundant but safe.
if 'vllm' not in ServerRegistry.list():
    ServerRegistry._registry['vllm'] = VLLMServer
    logger.info("Explicitly registered VLLMServer as 'vllm'")

# # If LMDeployServer uses @ServerRegistry.register('lmdeploy'), this is redundant but safe.
# if 'lmdeploy' not in ServerRegistry.list():
#     ServerRegistry._registry['lmdeploy'] = LMDeployServer
#     logger.info("Explicitly registered LMDeployServer as 'lmdeploy'")
