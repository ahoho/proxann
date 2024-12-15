# QWEN on WIKI

python3 run_user_study.py --model_type qwen:32b --prompt_mode q1_then_q3_dspy,q1_then_q2_dspy --removal_condition loose --path_save_results data/llm_out/wiki --config_path data/json_out/config_pilot_wiki.json,data/json_out/config_pilot_wiki_part2.json --response_csv data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv

# LLAMA ON WIKI

python3 run_user_study.py --model_type llama3.1:8b-instruct-q8_0 --prompt_mode q1_then_q3_dspy,q1_then_q2_dspy --removal_condition loose --path_save_results data/llm_out/wiki_new --config_path data/json_out/config_pilot_wiki.json,data/json_out/config_pilot_wiki_part2.json --response_csv data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv

# LLAMA ON BILLS

python3 run_user_study_bills.py --model_type llama3.1:8b-instruct-q8_0 --prompt_mode q1_then_q3_dspy,q1_then_q2_dspy --removal_condition loose --path_save_results data/llm_out/bills_examples_bills --config_path data/json_out/config_bills_part1.json --response_csv data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv

# GPT ON BILLS
python3 run_user_study_bills.py --model_type gpt-4o-2024-08-06 --prompt_mode q1_then_q3_dspy,q1_then_q2_dspy --removal_condition loose --path_save_results data/llm_out/bills_examples_bills --config_path data/json_out/config_bills_part1.json --response_csv data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv