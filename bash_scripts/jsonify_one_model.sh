#!/bin/bash

# FILE PATHS
SCRIPT="src/user_study_data_collector/jsonify/topic_json_formatter.py"
MODEL_PATH="data/models/mallet"
THETAS="data/models/mallet/thetas.npz"
BETAS="data/models/mallet/betas.npy"
VOCAB="data/training_data/bills/vocab.json"
CORPUS="data/training_data/bills/train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"
TEXT_COLUMN_DISP="summary"

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

# Function to check if a file exists
check_file() {
  if [ ! -f "$1" ]; then
    echo "Error: Required file $1 does not exist."
    exit 1
  fi
}

check_folder() {
  if [ ! -d "$1" ]; then
    echo "Error: Required folder $1 does not exist."
    exit 1
  fi
}

# Check if all required files exist
echo "Checking required files..."
check_folder "$MODEL_PATH"
check_file "$THETAS"
check_file "$BETAS"
check_file "$VOCAB"
check_file "$CORPUS"

# Run the Python script
echo "Running the jsonify script..."
python3 "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --thetas_path "$THETAS" \
    --betas_path "$BETAS" \
    --vocab_path "$VOCAB" \
    --corpus_path "$CORPUS" \
    --method "elbow" \
    --top_words 100 \
    --text_column_disp "$TEXT_COLUMN_DISP" #\
   # --trained_with_thetas_eval

# Check if the script succeeded
if [ $? -eq 0 ]; then
  echo "Script completed successfully."
else
  echo "Script failed. Please check the errors above."
  exit 1
fi