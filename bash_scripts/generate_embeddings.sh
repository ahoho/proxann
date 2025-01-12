#!/bin/bash

##############
# FILE PATHS #
##############
SCRIPT="src/embeddings/embedder.py"
SOURCE_FILE="data/training_data/bills/train.metadata.jsonl"
OUTPUT_FILE="data/training_data/bills/train.metadata.embeddings.parquet"
BATCH_SIZE=128
SBERT_MODEL="all-MiniLM-L6-v2"
AGGREGATE_EMBEDDINGS=False
CALCULATE_ON="summary"

# Go to project root
cd "$(dirname "$0")/.." || { echo "Failed to change directory."; exit 1; }

echo "Current working directory: $(pwd)"
if [ ! -f "$SCRIPT" ]; then
  echo "Error: $SCRIPT does not exists."
  exit 1
fi

# Activate venv
if [ -f ".venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
else
  echo "Error: Virtual environment not found at .venv. Please set it up."
  exit 1
fi

echo "Using Python version: $(python3 --version)"
echo "Python path: $(which python3)"

export PYTHONPATH=$(pwd)

echo "Running the embedding script..."
python3 "$SCRIPT" \
    --source_file "$SOURCE_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE" \
    --sbert_model "$SBERT_MODEL" \
    --aggregate_embeddings "$AGGREGATE_EMBEDDINGS" \
    --calculate_on "$CALCULATE_ON"

if [ $? -eq 0 ]; then
  echo "Embedding script completed successfully."
else
  echo "Embedding script failed. Please check the errors above."
  exit 1
fi
