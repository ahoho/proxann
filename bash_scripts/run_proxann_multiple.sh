#!/bin/bash
set -e

# Define common variables
MODEL_TYPE="meta-llama/Meta-Llama-3.1-8B-Instruct" #"gpt-4o-2024-08-06" #"qwen:32b" #"qwen:32b" #"llama3.3:70b" #"llama3.1:8b-instruct-q8_0" #,llama3.3:70b,qwen:32b
PROMPT_MODE="q1_then_q3_mean,q1_then_q2_mean"
REMOVAL_CONDITION="loose"
SAVE_PATH="camera_ready_llm_out/mean"
TEMPERATURES=1.0,0.0,0.0

# generate n random seeds
n=5
seeds=($(shuf -i 1-1000 -n $n))
#seeds=(338 436 499 742 853)
#seeds=(266)
echo "Seeds: ${seeds[@]}"

# Define dataset-specific configurations using arrays
DATASET_KEYS=("wiki" "bills")
TM_MODEL_DATA_PATHS=(
  "data/data_used_in_paper/json_out/config_wiki_part1.json,data/data_used_in_paper/json_out/config_wiki_part2.json"
  "data/data_used_in_paper/json_out/config_bills_part1.json,data/data_used_in_paper/json_out/config_bills_part2.json"
)
RESPONSE_CSV_PATHS=(
  "data/data_used_in_paper/qualtrics/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv"
  "data/data_used_in_paper/qualtrics/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv"
)

mkdir -p "$SAVE_PATH"

# Loop through datasets and run the command
for s in "${seeds[@]}"; do
  echo "Running for seed: $s"
  for i in "${!DATASET_KEYS[@]}"; do
    DATASET_KEY="${DATASET_KEYS[$i]}"
    TM_MODEL_DATA_PATH="${TM_MODEL_DATA_PATHS[$i]}"
    RESPONSE_CSV="${RESPONSE_CSV_PATHS[$i]}"

    echo "Running for dataset: $DATASET_KEY"

    SAVE_PATH_DTSET="$SAVE_PATH/$DATASET_KEY"

    mkdir -p "$SAVE_PATH_DTSET"

    echo python3 proxann_user_study.py \
      --model_type "$MODEL_TYPE" \
      --prompt_mode "$PROMPT_MODE" \
      --removal_condition "$REMOVAL_CONDITION" \
      --path_save_results "$SAVE_PATH_DTSET" \
      --tm_model_data_path "$TM_MODEL_DATA_PATH" \
      --response_csv "$RESPONSE_CSV" \
      --dataset_key "$DATASET_KEY" \
      --seed "$s" \
      --temperatures "$TEMPERATURES" \
      --do_both_ways

    python3 proxann_user_study.py \
      --model_type "$MODEL_TYPE" \
      --prompt_mode "$PROMPT_MODE" \
      --removal_condition "$REMOVAL_CONDITION" \
      --path_save_results "$SAVE_PATH_DTSET" \
      --tm_model_data_path "$TM_MODEL_DATA_PATH" \
      --response_csv "$RESPONSE_CSV" \
      --dataset_key "$DATASET_KEY" \
      --seed "$s" \
      --max_tokens 20 \
      --temperatures "$TEMPERATURES" \
      --do_both_ways
  done
done
