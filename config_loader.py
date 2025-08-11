import yaml
from typing import Dict, Any
import os

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary with validated configuration
        
    Raises:
        FileNotFoundError: If config file is missing
        ValueError: For invalid config values
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ["ollama", "embedding", "processing"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    
    # Set defaults for optional values
    config["processing"].setdefault("chunk_overlap", 64)
    config["processing"].setdefault("max_papers", 3)
    
    # Validate Ollama settings
    if not isinstance(config["ollama"].get("num_ctx"), int):
        config["ollama"]["num_ctx"] = 65536  # Default context window
    
    return config