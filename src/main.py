import argparse
from utils.helpers import load_config
from utils.config_validator import ConfigValidator
from data.dataset_loader import DatasetLoader
from data.data_processor import DataProcessor
from data.generate_qna import generate_qna
from model.model_loader import ModelLoader
from model.loss_functions import LossFunctionFactory
from training.trainer import Trainer
from training.evaluator import EvaluatorFactory
from config_loader import DATASET_CONFIGS
from typing import List, Tuple, Dict
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import clean_non_ascii_chars
from unstructured.cleaners.core import group_broken_paragraphs
from dotenv import load_dotenv 
import PyPDF2
import os
import pandas as pd 

def clean_data(text: str) -> str:
    text = replace_unicode_quotes(text)
    text = clean(text, bullets=True, lowercase=True, extra_whitespace=True,)
    text = clean_non_ascii_chars(text)
    text = group_broken_paragraphs(text)
    return text

def read_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            temp_text = page.extract_text()
            text += temp_text
            
    text = clean_data(text)

    return text

FILE_READERS = {
    '.pdf': read_pdf
}

def process_file(file_path: str) -> List[str]:
    _, file_extension = os.path.splitext(file_path)
    processor = FILE_READERS.get(file_extension.lower())  # Remove the dot from extension
    if processor:
        content = processor(file_path)
        return content
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_folder(folder_path: str) -> dict:
    results = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                results[filename] = process_file(file_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return results

def main(config_path: str = None, config_dict: dict = None):
    # Load and validate configuration
    if config_path:
        config = load_config(config_path)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    ConfigValidator.validate_config(config)

    input_folder = config['data']['input_folder']
    dataset_format = config['data']['dataset_format']

    folder_path = config['data']['input_folder']
    processed_files = process_folder(folder_path)

    data_processor = DataProcessor(config)
    content = data_processor.process_text(processed_files)    

    df = generate_qna(content, dataset_format)

    file_name = DATASET_CONFIGS[dataset_format]['file_name']
    df.to_csv(f"{input_folder}/{file_name}", index=False)

    # Load and process dataset
    dataset_loader = DatasetLoader(config)
    train_dataset = dataset_loader.get_train_dataset()
    eval_dataset = dataset_loader.get_eval_dataset()
    
    
    # Apply data processing if needed (e.g., for text chunking)
    # processed_dataset = dataset.map(lambda example: {"processed_text": data_processor.process_text(example["text"])})

    # Load model
    model_loader = ModelLoader(config)
    model = model_loader.load_model()

    # Create loss function
    loss_function = LossFunctionFactory.get_loss_function(config, model)

    evaluator = EvaluatorFactory.create_evaluator(config, eval_dataset) if eval_dataset else None

    # Create and run trainer
    trainer = Trainer(config, model, dataset_loader, loss_function, evaluator)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Model Fine-tuning Accelerator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    load_dotenv(override=True)
    main(config_path=args.config)

#
#def run_cli():
#    parser = argparse.ArgumentParser(description="Embedding Model Fine-tuning Accelerator")
#    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
#    args = parser.parse_args()
    
#    main(config_path=args.config)

#if __name__ == "__main__":
#    load_dotenv(override=True)
#    run_cli()