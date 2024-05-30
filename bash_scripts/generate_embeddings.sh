#!/bin/bash

cd "$(dirname "$0")/.."

pwd

python3 main.py generate_embeddings \
    --source_file "data/train.metadata.jsonl" \
    --output_file "data/train.metadata.enriched.parquet" \
    --batch_size 128 \
    --sbert_model "all-MiniLM-L6-v2" \
    --aggregate_embeddings False \
    --calculate_on "text"