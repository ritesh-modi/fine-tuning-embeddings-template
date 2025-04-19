import yaml
from typing import Dict, Any

def load_dataset_config() -> Dict[str, Any]:
    with open('config/platform_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['dataset_configs']

DATASET_CONFIGS = load_dataset_config()