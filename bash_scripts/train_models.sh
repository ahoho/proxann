#!/bin/bash

CONFIG_FILE="src/proxann/config/config.yaml"
TRAINER_TYPE="TomotopyLda"
NUM_TOPICS=50
SCRIPT="python3 -m proxann.topic_models.train.tm_trainer"

cd "$(dirname "$0")/.."

export PYTHONPATH=$(pwd)

DATASETS=(
    "bills:data/training_data/bills"
    #"wikitext:/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/wikitext"
)

MODEL_BASE_PATH="data/test/malet"
VOCAB_FILE="vocab.json"
EMBEDDINGS_SUFFIX="train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"

# Iterate through the array
for DATASET_ENTRY in "${DATASETS[@]}"; do
    # Split the dataset entry into name and path
    IFS=':' read -r DATASET DATASET_PATH <<< "$DATASET_ENTRY"
    
    # Construct paths
    CORPUS_FILE="${DATASET_PATH}/${EMBEDDINGS_SUFFIX}"
    MODEL_PATH="${MODEL_BASE_PATH}/${DATASET}-labeled/vocab_15k/k-${NUM_TOPICS}/new_tests_bertopic"
    VOCAB_PATH="${DATASET_PATH}/${VOCAB_FILE}"

    # Output current processing dataset
    echo "Training model for dataset: $DATASET"

    # Run the training script
    $SCRIPT \
        --config_path "$CONFIG_FILE" \
        --corpus_file "$CORPUS_FILE" \
        --model_path "$MODEL_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --trainer_type "$TRAINER_TYPE" \
        --num_topics "$NUM_TOPICS"
done