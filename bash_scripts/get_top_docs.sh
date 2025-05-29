#!/bin/bash

##############
# FILE PATHS #
##############
SCRIPT="src/user_study_data_collector/topics_docs_selection/doc_selector.py"
MODEL_PATH="data/models/mallet"
CORPUS_PATH="data/training_data/bills/train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"
METHOD="thetas,thetas_sample"
TOP_WORDS=100
TRAINED_WITH_THETAS_EVAL=True

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

echo "Running the doc selector script..."
if [ "$TRAINED_WITH_THETAS_EVAL" = True ]; then
    python3 "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --method "$METHOD" \
    --top_words "$TOP_WORDS" \
    --trained_with_thetas_eval
else
    python3 "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --method "$METHOD" \
    --top_words "$TOP_WORDS"
fi

if [ $? -eq 0 ]; then
  echo "Doc selector script completed successfully."
else
  echo "Doc selector script failed. Please check the errors above."
  exit 1
fi