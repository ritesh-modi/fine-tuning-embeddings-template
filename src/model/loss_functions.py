from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss, TripletLoss, ContrastiveLoss, CosineSimilarityLoss
from typing import Dict, Any

class LossFunctionFactory:
    @staticmethod
    def get_loss_function(config: Dict[str, Any], model):
        loss_function = config['training']['loss_function']
        
        if loss_function == 'matryoshka':
            inner_loss = MultipleNegativesRankingLoss(model)
            return MatryoshkaLoss(model, inner_loss, matryoshka_dims=config['training']['matryoshka_dimensions'])
        elif loss_function == 'triplet':
            return TripletLoss(model)
        elif loss_function == 'contrastive':
            return ContrastiveLoss(model)
        elif loss_function == 'cosine_similarity':
            return CosineSimilarityLoss(model)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")