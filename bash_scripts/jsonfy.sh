#!/bin/bash

cd "$(dirname "$0")/.."

pwd

python3 main.py jsonfy \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/mallet_wiki_50" \
    --corpus_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.jsonl" \
    --method "thetas" \
    --top_words 100 \
    --trained_with_thetas_eval True
