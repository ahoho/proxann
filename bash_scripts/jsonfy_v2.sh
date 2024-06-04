#!/bin/bash

cd "$(dirname "$0")/.."

pwd

python3 main.py jsonfy \
    --thetas_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/mallet/doctopics.npz.npy" \
    --betas_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/mallet/beta.npy" \
    --vocab_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/mallet/vocab.json" \
    --corpus_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.jsonl" \
    --method "thetas" \
    --top_words 100 