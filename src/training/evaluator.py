from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator, TripletEvaluator, SentenceEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.similarity_functions import SimilarityFunction
from typing import Dict, Any
import numpy as np

class EvaluatorFactory:
    @staticmethod
    def create_evaluator(config: Dict[str, Any], dataset):
        if config['data']['dataset_format'] == 'positive_pair':
            return EvaluatorFactory._create_ir_evaluator(config, dataset)
        elif config['data']['dataset_format'] == 'triplets':
            return EvaluatorFactory._create_triplet_evaluator(config, dataset)
        elif config['data']['dataset_format'] == 'pair_with_score':
            return EvaluatorFactory._create_pair_with_score_evaluator(config, dataset)
        else:
            raise NotImplementedError(f"Evaluator not implemented for dataset format: {config['data']['dataset_format']}")

    @staticmethod
    def _create_ir_evaluator(config: Dict[str, Any], dataset) -> SentenceEvaluator:
        corpus = dict(zip(range(len(dataset)), dataset['positive']))
        queries = dict(zip(range(len(dataset)), dataset['anchor']))
        relevant_docs = {q_id: [q_id] for q_id in queries.keys()}

        matryoshka_evaluators = []
        for dim in config['training']['matryoshka_dimensions']:
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"dim_{dim}",
                truncate_dim=dim,
                score_functions={"cosine": cos_sim},
            )
            matryoshka_evaluators.append(ir_evaluator)

        return SequentialEvaluator(matryoshka_evaluators)
    
    @staticmethod
    def _create_triplet_evaluator(config: Dict[str, Any], dataset) -> SentenceEvaluator:
        # For triplets, we'll use the TripletEvaluator
        return TripletEvaluator(dataset['anchor'], dataset['positive'], dataset['negative'])

    @staticmethod
    def _create_pair_with_score_evaluator(config: Dict[str, Any], dataset) -> SentenceEvaluator:
        sentences1 = []
        sentences2 = []
        scores = []

        for s1, s2, score in zip(dataset['sentence1'], dataset['sentence2'], dataset['score']):
            if score is not None:
                try:
                    float_score = float(score)
                    sentences1.append(s1)
                    sentences2.append(s2)
                    scores.append(float_score)
                except ValueError:
                    print(f"Warning: Could not convert score '{score}' to float. Skipping this pair.")

        if not scores:
            raise ValueError("No valid similarity scores found in the dataset.")

        # Convert scores to a numpy array
        scores = np.array(scores)

        # Normalize scores to be between 0 and 1 if they aren't already
        if np.max(scores) > 1 or np.min(scores) < 0:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        # Create the evaluator
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1,
            sentences2,
            scores,
            name="pair_similarity_evaluation",
            main_similarity=SimilarityFunction.COSINE
        )

        return evaluator