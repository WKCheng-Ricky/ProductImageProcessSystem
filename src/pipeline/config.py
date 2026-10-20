"""
Configuration Management

Handles environment variables and configuration for the application.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# Pipeline defaults (single source of truth; values match the previous scattered literals)
DEFAULT_TARGET_SIZE = (512, 512)
DEFAULT_OUTPUT_DIR = "uploads/processed"
PIPELINE_SIMILARITY_THRESHOLD = 0.85
GROUPING_SIMILARITY_THRESHOLD = 0.90
DUPLICATE_SIMILARITY_THRESHOLD = 0.999
QUALITY_THRESHOLD = 0.3
GROUPING_TOP_K = 10

# CLIP model defaults
DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"
# The aesthetic scorer was tuned on LAION weights, so its fallback differs on purpose.
AESTHETIC_FALLBACK_MODEL = "ViT-B-32"
AESTHETIC_FALLBACK_PRETRAINED = "laion2b_s34b_b79k"


def str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    return value.lower() in ('true', '1', 'yes', 'on')


def clip_model_from_env() -> Optional[str]:
    return os.getenv("CLIP_MODEL")


def clip_pretrained_from_env() -> Optional[str]:
    return os.getenv("CLIP_PRETRAINED")


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
