# Sentiment-Aware Movie Review Finder

![Notebook](https://img.shields.io/badge/notebook-Jupyter-orange) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Table of contents

- [About](#about)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook overview](#notebook-overview)
- [Datasets & resources](#datasets--resources)
- [Results (sample)](#results-sample)
- [How it works (brief)](#how-it-works-brief)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

This project provides an interactive Jupyter Notebook titled **Sentiment Aware Movie Review Finder**. It 
demonstrates how to:
- Load and preprocess movie review data.
- Build or use sentiment analysis models to infer review polarity.
- Index and search reviews with sentiment-aware ranking.
- Visualize results and evaluate retrieval and sentiment detection performance.

## Features

- Notebook-based demonstration: exploratory data analysis, modeling, and evaluation.
- Sentiment-aware retrieval: rank or filter search results by predicted sentiment.
- Reproducible preprocessing and evaluation pipelines.
- Visualizations for model performance and sample predictions.

## Repository structure

Files included in the repository workspace (on this environment):

- `Sentiment Aware Movie Review Finder.ipynb`

## Requirements

The notebook uses the following Python packages (detected from import statements). Install these with `pip` or `conda` as needed:

- `faiss`
- `matplotlib`
- `numpy`
- `pandas`
- `pickle`
- `plotly`
- `pyngrok`
- `seaborn`
- `sentence_transformers`
- `sklearn`
- `streamlit`
- `torch`
- `tqdm`
- `transformers`

Recommended base environment:

```bash
python>=3.8
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. (Optional) Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

3. Launch the notebook:

```bash
jupyter notebook "Sentiment Aware Movie Review Finder.ipynb"
```

## Usage

Open the notebook in Jupyter Lab/Notebook or execute it cell-by-cell. Key sections include data loading, preprocessing, model training/inference, indexing/search, and visualization. Typical workflows:
1. Prepare dataset (place CSV/JSON files in `data/` or adjust file paths in cells).
2. Run preprocessing cells to clean and tokenize text.
3. Train or load a pre-trained sentiment model.
4. Build the search index and execute queries.
5. Inspect visualizations and evaluation metrics.

## Notebook overview

Detected notebook sections (top-level headings):

- # Cell 1: Environment Setup & Installations
- # Cell 2: Load and Explore Dataset
- # Cell 3: Data Preprocessing
- # Cell 4: Generate Embeddings
- # Cell 5: Build FAISS Indexes
- # Cell 6: Sentiment Classifier for Query
- # Cell 7A: Retrieval Pipeline WITHOUT Sentiment (Pure Semantic Search)
- # Cell 7B: Retrieval Pipeline WITH Sentiment (Sentiment-Aware Search)
- # Cell 8: Evaluation Metrics Implementation
- # Cell 9: Ablation Study - Detailed Analysis
- # Cell 10A: Query Testing Without Sentiment
- # Cell 10B: Query Testing With Sentiment
- # Cell 11: Streamlit Dashboard

## Datasets & resources

The notebook references the following data or resource files:

- `/content/IMDB Dataset.csv`
- `reviews_processed.csv`

## Results (sample)

Some representative snippets are shown below (truncated):

```
All packages installed successfully
FAISS version: 1.13.2
SentenceTransformers version: 5.2.0
```

```
Dataset loaded successfully from: /content/IMDB Dataset.csv
Dataset Shape: (50000, 2)
```

## How it works (brief)

- Use SentenceTransformers to encode reviews into embeddings.
- Build FAISS indexes (FlatL2, IVF, HNSW, IVF+PQ) over normalized embeddings.
- Use a DistilBERT-based sentiment classifier to infer query sentiment and optionally filter results.
- Evaluate retrieval quality using mAP, MRR, Precision@k, Recall@k.

## Contributing

Contributions are welcome. Please open issues or pull requests for improvements.

## License

Specify a license if you have one (e.g., MIT). If none, add one to clarify reuse.

## Contact

Project author and maintainer: `Zahin2470` (your GitHub username)

