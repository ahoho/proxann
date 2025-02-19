#!/bin/bash

# Define common variables
MODEL_TYPE="qwen:32b" #"gpt-4o-2024-08-06" #"qwen:32b" #"qwen:32b" #"llama3.3:70b" #"llama3.1:8b-instruct-q8_0" #,llama3.3:70b,qwen:32b
PROMPT_MODE="q1_then_q3_dspy,q1_then_q2_dspy"
REMOVAL_CONDITION="loose"
SAVE_PATH="data/tests"

# Define dataset-specific configurations using arrays
DATASET_KEYS=("wiki" "bills") #"bills" 
TM_MODEL_DATA_PATHS=(
  "data/json_out/arr_dec/config_pilot_wiki.json,data/json_out/arr_dec/config_pilot_wiki_part2.json"
  "data/json_out/arr_dec/config_pilot_bills.json"
)
RESPONSE_CSV_PATHS=(
  "data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv"
  "data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv"
)

mkdir -p "$SAVE_PATH"

# Loop through datasets and run the command
for i in "${!DATASET_KEYS[@]}"; do
  DATASET_KEY="${DATASET_KEYS[$i]}"
  TM_MODEL_DATA_PATH="${TM_MODEL_DATA_PATHS[$i]}"
  RESPONSE_CSV="${RESPONSE_CSV_PATHS[$i]}"

  echo "Running for dataset: $DATASET_KEY"

  SAVE_PATH_DTSET="data/arr_dec_responses/$DATASET_KEY"

  mkdir -p "$SAVE_PATH_DTSET"

  python3 proxann_user_study.py \
    --model_type "$MODEL_TYPE" \
    --prompt_mode "$PROMPT_MODE" \
    --removal_condition "$REMOVAL_CONDITION" \
    --path_save_results "$SAVE_PATH_DTSET" \
    --tm_model_data_path "$TM_MODEL_DATA_PATH" \
    --response_csv "$RESPONSE_CSV" \
    --dataset_key "$DATASET_KEY"
done