from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

from typing import Dict, Any
import os

class Trainer:
    def __init__(self, config: Dict[str, Any], model: SentenceTransformer, dataset_loader, loss_function, evaluator):
        self.config = config
        self.model = model
        self.dataset_loader = dataset_loader 
        self.loss_function = loss_function
        self.evaluator = evaluator

    def train(self):
        args = self._get_training_arguments()

        train_dataset = self.dataset_loader.get_train_dataset()
        eval_dataset = self.dataset_loader.get_eval_dataset()

        if self.evaluator:
            self.model.eval()
            dummy_scores = self.evaluator(self.model, output_path=None)
            best_metric = self._choose_best_metric(dummy_scores)
            args.metric_for_best_model = best_metric
        
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=self.loss_function,
            evaluator=self.evaluator if eval_dataset is not None else None,
        )



        trainer.train()
        trainer.save_model()

    def _get_training_arguments(self):
        return SentenceTransformerTrainingArguments(
            output_dir=os.path.join(self.config['model']['output_dir'], self.config['data']['dataset_format']),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            per_device_eval_batch_size=self.config['training']['batch_size'] * 2,
            warmup_ratio=self.config['training']['warmup_ratio'],
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            tf32=False,
            bf16=True,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy=self.config['evaluation']['strategy'],
            save_strategy=self.config['evaluation']['strategy'],
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            #metric_for_best_model=f"eval_dim_128_cosine_{self.config['evaluation']['metric']}",
        )
    
    @staticmethod
    def _choose_best_metric(metrics):
        # Priority order for choosing the best metric
        priority_metrics = ['eval_cosine_accuracy', 'eval_dot_accuracy', 'eval_manhattan_accuracy', 
                            'eval_euclidean_accuracy', 'eval_max_accuracy', 'eval_loss']
        
        for metric in priority_metrics:
            if metric in metrics:
                return metric
        
        # If none of the priority metrics are found, return the first available metric
        return next(iter(metrics))      