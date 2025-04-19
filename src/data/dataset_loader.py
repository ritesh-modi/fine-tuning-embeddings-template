import logging
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from config_loader import DATASET_CONFIGS

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_folder = config['data']['input_folder']
        self.dataset_format = config['data']['dataset_format']
        self.train_ratio = 0.7
        self.eval_ratio = 0.3
        self._dataset = None

    def load_dataset(self) -> DatasetDict:
        if self._dataset is None:
            try:
                dataset = self._load_dataset_by_format()
                self._dataset = self._split_dataset(dataset)
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise
        return self._dataset

    def _load_dataset_by_format(self) -> Dataset:
        file_name = DATASET_CONFIGS[self.dataset_format]['file_name']
        return load_dataset('csv', data_files=f"{self.input_folder}/{file_name}")['train']

    def get_train_dataset(self) -> Dataset:
        return self.load_dataset()['train']

    def get_eval_dataset(self) -> Dataset:
        return self.load_dataset()['validation']

    def _split_dataset(self, dataset: Dataset) -> DatasetDict:
        train_testvalid = dataset.train_test_split(test_size=self.eval_ratio, seed=42)
        return DatasetDict({
            'train': train_testvalid['train'],
            'validation': train_testvalid['test']
        })

    def _load_positive_pair_dataset(self) -> Dataset:
        dataset = load_dataset('csv', data_files=f"{self.input_folder}/positive_pairs.csv")
        return dataset.map(lambda example, idx: {
            "anchor": example["Question"],
            "positive": example["Answer"]
        }, with_indices=True, remove_columns=["Question", "Answer"])

    def _load_triplets_dataset(self) -> Dataset:
        dataset = load_dataset('csv', data_files=f"{self.input_folder}/triplets.csv")
        return dataset.map(lambda example, idx: {
            "anchor": example["anchor"],
            "positive": example["positive"],
            "negative": example["negative"]
        }, with_indices=True)

    def _load_pair_with_score_dataset(self) -> Dataset:
        dataset = load_dataset('csv', data_files=f"{self.input_folder}/pairs_with_scores.csv")
        return dataset.map(lambda example, idx: {
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "similarity_score": float(example["similarity_score"])
        }, with_indices=True)

    def _load_texts_with_classes_dataset(self) -> Dataset:
        dataset = load_dataset('csv', data_files=f"{self.input_folder}/texts_with_classes.csv")
        return dataset.map(lambda example, idx: {
            "text": example["text"],
            "class": example["class"]
        }, with_indices=True)