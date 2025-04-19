import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

class ModelLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_model = config['model']['base_model']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        return SentenceTransformer(self.base_model, device=self.device)