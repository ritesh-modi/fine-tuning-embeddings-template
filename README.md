# Fine-Tuning Embeddings Template

A template repository for fine-tuning sentence embedding models using the [`sentence-transformers`](https://www.sbert.net/) package. This project supports fine-tuning on various types of datasets and loss functions, including synthetic dataset generation from PDF files using Azure OpenAI.

## 🚀 Features

- Fine-tune any SentenceTransformer-compatible model
- Supports multiple dataset formats:
  - `positive_pair`
  - `triplets`
  - `pair_with_score`
- Generates synthetic training data from PDFs using Azure OpenAI
- Compatible with various loss functions:
  - `matryoshka`
  - `triplet`
  - `contrastive`
  - `cosine_similarity`
- Evaluates the fine-tuned model using metrics such as NDCG@k
- CLI and config file-based execution
- Outputs fine-tuned models and evaluation results

---

## 📁 Folder Structure

```
.
├── config/               # Configuration files for fine-tuning and evaluation
├── data/                 # Place your PDF files here for data generation
├── output/               # Fine-tuned models and evaluation metrics
├── src/                  # Core source code
├── main.py               # Entry point for training/evaluation
├── .env                  # Azure OpenAI secrets
└── requirements.txt      # Dependencies
```

---

## 🔧 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ritesh-modi/fine-tuning-embeddings-template.git
cd fine-tuning-embeddings-template
```

### 2. Create a Python Environment

Using conda (recommended):

```bash
conda create -n fine-tune-env python=3.10
conda activate fine-tune-env
```

### 3. Install Dependencies in Editable Mode

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

### 4. Set Azure OpenAI Secrets

Create a `.env` file in the root directory:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
API_VERSION=
AZURE_DEPLOYMENT=
MODEL_NAME=
TEMPERATURE=0.0
```

---

## 📘 Usage Instructions

### Step 1: Add Your PDF Files

Place your PDF documents inside the `data/` directory. The repo will generate synthetic training data from these using Azure OpenAI.

### Step 2: Update Configuration

Modify the configuration files inside the `config/` folder to specify:

- Dataset type (`positive_pair`, `triplets`, `pair_with_score`)
- Model to fine-tune
- Loss function
- Output paths

⚠️ **Note:** Not all dataset types are compatible with every loss function. Refer to the official [`sentence-transformers`](https://www.sbert.net/) documentation for compatibility.

Example:
- `triplets` dataset → `TripletLoss`
- `positive_pair` → `CosineSimilarityLoss`, etc.

### Step 3: Run Fine-Tuning

```bash
python main.py --config_path config/train_config.yaml
```

This will:

- Generate synthetic training data (if configured)
- Load and preprocess the data
- Train the model with the selected loss function
- Evaluate the fine-tuned model and store results in the `output/` directory

---

## 📊 Evaluation

Evaluation metrics like **NDCG@k**, **MAP**, and **Recall** are automatically calculated post-training. The fne-tuned model will be saved in the `output/` directory.

---

## 🧠 Dataset & Loss Function Compatibility

Ensure the dataset format is compatible with the selected loss function. Here are some guidelines:

| Dataset Type     | Supported Loss Functions                  |
|------------------|--------------------------------------------|
| `triplets`       | `TripletLoss`                              |
| `positive_pair`  | `CosineSimilarityLoss`, `ContrastiveLoss` |
| `pair_with_score`| `MatryoshkaLoss`, `CosineSimilarityLoss`  |

📖 Visit the [SentenceTransformers Loss Functions Docs](https://www.sbert.net/docs/package_reference/losses.html) for detailed compatibility.

---

## 📌 Requirements

- Python 3.10+
- Conda (optional but recommended)
- Azure OpenAI account with deployment

Install all dependencies using:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgments

- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview)