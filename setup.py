from setuptools import setup, find_packages

setup(
    name="embedding-finetuning-accelerator",
    version="0.1.0",

    install_requires=[
        "torch",
        "sentence-transformers",
        "datasets",
        "pyyaml",
        "nltk",
    ],
    entry_points={
        "console_scripts": [
            "run-finetuning=src.main:run_cli",
        ],
    },
    author="Ritesh Modi",
    author_email="ritesh.modi@outlook.com",
    description="An accelerator for fine-tuning embedding models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages = ["src"],
    url="https://github.com/ritesh-modi/fine-tuning-embeddings-template",
)