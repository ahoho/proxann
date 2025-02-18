"""
Main script for evaluating a topic using the Proxann model.

Usage:
------

1. If the model has been trained using Proxann, only the paths to the trained model and the dataset are required. Set `trained_with_thetas_eval=True`.
    
Example:
--------
model_path = "data/models/mallet"
corpus_path = "data/training_data/bills/train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"
trained_with_thetas_eval = True

2. If the model was trained separately, you must specify additional paths:
- Thetas file (as a sparse matrix)
- Betas file
- Vocabulary file (JSON)``
- Corpus file
- Set `trained_with_thetas_eval=False`

Example:
--------
mallet_config = {
"thetas_path": "data/models/mallet/doctopics.npz.npy",
"betas_path": "data/models/mallet/beta.npy",
"vocab_path": "data/models/mallet/vocab.json",
"corpus_path": "data/train.metadata.jsonl",
"trained_with_thetas_eval": False
}

Ensure that all required files are correctly formatted and accessible.
"""

# @TODO: Make json with user-provided model data
# @TODO: Run Q1, Q2, Q3 and get a score based on correlations with the topic model
