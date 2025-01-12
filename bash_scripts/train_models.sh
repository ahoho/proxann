#!/bin/bash

TRAINER_TYPE="BERTopic"
NUM_TOPICS=50
SCRIPT="python3 src/train/tm_trainer.py"

cd "$(dirname "$0")/.."

export PYTHONPATH=$(pwd)

DATASETS=(
    "bills:data/training_data/bills"
    #"wikitext:/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/wikitext"
)

MODEL_BASE_PATH="data/models/final_models/model-runs"
VOCAB_FILE="vocab.json"
EMBEDDINGS_SUFFIX="train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"

# Iterate through the array
for DATASET_ENTRY in "${DATASETS[@]}"; do
    # Split the dataset entry into name and path
    IFS=':' read -r DATASET DATASET_PATH <<< "$DATASET_ENTRY"
    
    # Construct paths
    CORPUS_FILE="${DATASET_PATH}/${EMBEDDINGS_SUFFIX}"
    MODEL_PATH="${MODEL_BASE_PATH}/${DATASET}-labeled/vocab_15k/k-${NUM_TOPICS}/bertopic"
    VOCAB_PATH="${DATASET_PATH}/${VOCAB_FILE}"

    # Output current processing dataset
    echo "Training model for dataset: $DATASET"

    # Run the training script
    $SCRIPT \
        --corpus_file "$CORPUS_FILE" \
        --model_path "$MODEL_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --trainer_type "$TRAINER_TYPE" \
        --num_topics "$NUM_TOPICS"
done