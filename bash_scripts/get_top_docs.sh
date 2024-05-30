#!/bin/bash

cd "$(dirname "$0")/.."

pwd

python3 main.py get_top_docs \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/mallet_wiki_50" \
    --corpus_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.jsonl" \
    --method "thetas,thetas_sample,sall,spart" \
    --top_words 100 \
    --trained_with_thetas_eval True
