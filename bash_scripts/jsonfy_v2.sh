#!/bin/bash

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

# Current working directory
pwd

# Check if the paths exist
if [ ! -f "data/models/mallet/doctopics.npz.npy" ]; then
  echo "The file data/models/mallet/doctopics.npz.npy does not exist."
  exit 1
fi

if [ ! -f "data/models/mallet/beta.npy" ]; then
  echo "The file data/models/mallet/beta.npy does not exist."
  exit 1
fi

if [ ! -f "data/models/mallet/vocab.json" ]; then
  echo "The file data/models/mallet/vocab.json does not exist."
  exit 1
fi

if [ ! -f "data/train.metadata.jsonl" ]; then
  echo "The file data/train.metadata.jsonl does not exist."
  exit 1
fi

python3 main.py jsonfy \
    --thetas_path "data/models/mallet/doctopics.npz.npy" \
    --betas_path "data/models/mallet/beta.npy" \
    --vocab_path "data/models/mallet/vocab.json" \
    --corpus_path "data/train.metadata.jsonl" \
    --method "thetas" \
    --top_words 100