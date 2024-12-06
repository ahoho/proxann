#!/bin/bash

TRAINER_TYPE="BERTopic"
NUM_TOPICS=50
SCRIPT="python3 main.py train_tm"

declare -A DATASETS
DATASETS["bills"]="/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/bills"
DATASETS["wikitext"]="/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/wikitext"

MODEL_BASE_PATH="/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/final_models/model-runs"
VOCAB_FILE="vocab.json"
EMBEDDINGS_SUFFIX="train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"

echo "${!DATASETS[@]}"

for DATASET in "${!DATASETS[@]}"; do
    CORPUS_FILE="${DATASETS[$DATASET]}/${EMBEDDINGS_SUFFIX}"
    MODEL_PATH="${MODEL_BASE_PATH}/${DATASET}-labeled/vocab_15k/k-${NUM_TOPICS}/bertopic"
    VOCAB_PATH="${DATASETS[$DATASET]}/${VOCAB_FILE}"

    echo "Training model for dataset: $DATASET"
    $SCRIPT \
        --corpus_file "$CORPUS_FILE" \
        --model_path "$MODEL_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --trainer_type "$TRAINER_TYPE" \
        --num_topics "$NUM_TOPICS"
done
