from typing import Dict, Any

class ConfigValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]):
        dataset_format = config['data']['dataset_format']
        loss_function = config['training']['loss_function']
        
        valid_combinations = {
            'positive_pair': ['matryoshka', 'contrastive'],
            'triplets': ['triplet'],
            'pair_with_score': ['cosine_similarity'],
            'texts_with_classes': ['matryoshka']
        }
        
        if dataset_format not in valid_combinations:
            raise ValueError(f"Invalid dataset format: {dataset_format}")
        
        if loss_function not in valid_combinations[dataset_format]:
            raise ValueError(f"Invalid loss function '{loss_function}' for dataset format '{dataset_format}'. Valid options are: {valid_combinations[dataset_format]}")
        
        # Add more validation checks as needed