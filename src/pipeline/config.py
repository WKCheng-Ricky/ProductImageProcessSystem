"""
Configuration Management

Handles environment variables and configuration for the application.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    return value.lower() in ('true', '1', 'yes', 'on')


def get_config() -> Dict[str, Any]:
    """
    Get application configuration from environment variables

    Returns:
        Dictionary containing configuration values
    """
    return {
        # Vector Database Configuration
        "use_milvus": str_to_bool(os.getenv("USE_MILVUS", "false")),
        "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
        "milvus_port": int(os.getenv("MILVUS_PORT", "19530")),
        "milvus_lite_path": os.getenv("MILVUS_LITE_PATH", "./data/milvus.db"),
        "milvus_collection": os.getenv("MILVUS_COLLECTION", "product_embeddings"),
    }


# Global config instance (loaded once)
CONFIG = get_config()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Current Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
