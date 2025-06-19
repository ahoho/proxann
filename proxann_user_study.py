import argparse
from collections import Counter, defaultdict
import os
import datetime

import pandas as pd

from src.proxann.prompter import Prompter
from src.proxann.proxann import ProxAnn
from src.proxann.utils import (
    collect_fit_rank_data, load_config_pilot, normalize_key, process_responses
)
from src.utils.utils import init_logger, load_yaml_config_file, log_or_print

Q1_THEN_Q2_PROMPTS = {"q1_then_q2_mean"}
Q1_THEN_Q3_PROMPTS = {"q1_then_q3_mean"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str,  required=False, default="config/config.yaml",
        help="Path to the configuration file.")
    parser.add_argument(
        "--running_mode", type=str, default="run", required=False,
        help="Running mode: run or eval.")
    parser.add_argument(
        "--dataset_key", type=str, default="wiki",
        help="Dataset to use for filtering the data.")
    parser.add_argument(
        "--model_type", type=str, default="llama3.2",
        help="LLM types to evaluate, separated by commas (e.g., llama3.2,llama3.1:8b-instruct-q8_0).")
    parser.add_argument(
        "--prompt_mode", type=str, default="q1_then_q3_dspy,q1_then_q2_dspy",
        help="Prompting modes, separated by commas (e.g., q1_then_q3_dspy,q1_then_q2_dspy).")
    parser.add_argument(
        "--tm_model_data_path", type=str,
        help="Path to JSON config files with model data (from user_study_data_collector).",
        default="data/files_pilot/config_first_round.json,data/files_pilot/config_second_round.json")
    parser.add_argument(
        "--response_csv", type=str,
        help="Path to the CSV with human responses from Qualtrics.",
        default="data/files_pilot/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv")
    parser.add_argument(
        "--do_both_ways", action="store_true",
        help="Run Q3 twice: once with A as the first document, then reversed.")
    parser.add_argument(
        "--use_user_cats", action="store_true",
        help="Use user categories for Q2/Q3 instead of LLM-generated ones from Q1.",
        default=False)
    parser.add_argument(
        "--removal_condition", type=str,
        default="loose",
        help="Condition for disqualifying responses ('loose': 1+ failures, 'strict': all failures)."
    )
    parser.add_argument(
        "--path_save_results", type=str,
        help="Path to save results.",
        default="data/files_pilot/results")
    parser.add_argument(
        "--temperatures", type=str, default=None,
        help="Temperatures value for the LLM generation in Q1/Q2/Q3, separated by commas."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for random number generator." 
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max tokens for the LLM generation."
    )
    return parser.parse_args()


def save_results(data, path, filename):
    """Save data to a JSON file."""
    # Check if data exists and is a DataFrame.
    # Handles cases where Q2 or Q3 results are not generated, e.g., when the script runs for a single mode instead of the default.
    if data is None or len(data) == 0:
        print(f"No data to save for {filename}")
        return
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_json(os.path.join(path, filename), orient="records")

    assert os.path.exists(os.path.join(path, filename)), f"Error saving {filename}"

def main():
    args = parse_args()

    # Init logger and load config
    logger = init_logger(args.config_path, f"RunProxann-{args.running_mode}")
    logger.info(f"Running Proxann in mode: {args.running_mode}")
    config = load_yaml_config_file(args.config_path, "user_study", logger)
    
    # Init proxann object
    proxann = ProxAnn(logger, args.config_path)
    
    # Get seed and temperature if given
    custom_temperatures = [float(el) for el in args.temperatures.split(",")] if args.temperatures is not None else None
    q1_temp = custom_temperatures[0] if custom_temperatures else 0
    q2_temp = custom_temperatures[1] if custom_temperatures else 0
    q3_temp = custom_temperatures[2] if custom_temperatures else 0
    print(f"Temperatures: {q1_temp}, {q2_temp}, {q3_temp}")
    custom_seed = args.seed if args.seed is not None else None
    custom_max_tokens = args.max_tokens if args.max_tokens is not None else None
        
    valid_models = config.get(
        "valid_models", {"mallet", "ctm", "bertopic", "category-45"})
    valid_datasets = config.get(
        "valid_datasets", {"wikitext-labeled", "bills-labeled"})

    # Load topic modeling data with information for each topic being evaluated and normalize the keys
    tm_model_data = load_config_pilot(args.tm_model_data_path)
    tm_model_data = {
        normalize_key(key, valid_models, valid_datasets, False): value for key, value in tm_model_data.items()
    }

    # Parse user's responses
    responses_by_id = process_responses(
        args.response_csv, args.tm_model_data_path.split(","), removal_condition=args.removal_condition, filter=args.dataset_key)

    # Get correlations
    _, _, _, corr_data = collect_fit_rank_data(responses_by_id)
    corr_data = sorted(corr_data, key=lambda x: x["id"])

    model_types = args.model_type.split(",") if args.model_type else []
    prompt_modes = args.prompt_mode.split(",") if args.prompt_mode else []

    llm_results_q1, llm_results_q2, llm_results_q3, all_info_bradley_terry = [], [], [], []
    topics_per_model = Counter()

    for prompt_mode in prompt_modes:
        log_or_print(f"Executing in MODE: {prompt_mode}", logger)

        # ---------------------------------------------------------
        # For each topic model (mallet / ctm / bertopic) ...
        # ---------------------------------------------------------
        # this selects the set of topics for a given model (mallet, ctm, etc.)
        for model_id, model_data in tm_model_data.items():
            log_or_print(f"Model: {model_id}", logger)

            # ---------------------------------------------------------
            # For each topic ...
            # ---------------------------------------------------------
            # each cluster_data is the information for a topic
            for cluster_id, cluster_data in model_data.items():
                log_or_print(f"Cluster: {cluster_id}", logger)
                # topic information
                id_ = f"{model_id}/{cluster_id}"
                model = model_id.split("/")[-1]
                topics_per_model[model_id] += 1
                topic_match_id = topics_per_model[model_id]

                # user data (categories, ranks)
                # users_cats = []
                # users_rank = []
                users_cats = [user_data["category"] for user_data in responses_by_id[id_]]
                this_corr_data = next(c for c in corr_data if c["id"] == id_)
                users_rank = this_corr_data["rank_data"]

                # ---------------------------------------------------------
                # For each LLM ...
                # ---------------------------------------------------------
                # to store the rank data for each lmm
                rank_data = []  # it will store the rank data for each lmm
                info_to_bradley_terry = defaultdict(list)  # it will store the raw rankings pairs that are passed to the Bradley-Terry model
                fit_data = []  # it will store the fit data for each lmm
                categories = []  # it will store the categories for each lmm

                for llm_model in model_types:
                    log_or_print(f"LLM: {llm_model}", logger)
                    prompter = Prompter(model_type=llm_model, seed=custom_seed, max_tokens=custom_max_tokens)

                    if prompt_mode in Q1_THEN_Q3_PROMPTS:
                        log_or_print("-- Executing Q1 / Q3...", logger)

                        # ==============================================
                        # Q1
                        # ==============================================
                        if prompt_mode in Q1_THEN_Q3_PROMPTS:
                            proxann.do_q1(
                                prompter=prompter, 
                                cluster_data=cluster_data, 
                                users_cats=users_cats, 
                                categories=categories, 
                                temperature=q1_temp
                            )
                            category = categories[-1]
                        else:
                            category = None

                        # ==============================================
                        # Q3
                        # ==============================================
                        # TODO: Add logic for when category is not category[-1] but each of the users' categories
                        proxann.do_q3(
                            prompter=prompter,
                            prompt_mode=prompt_mode,
                            cluster_data=cluster_data,
                            rank_data=rank_data,
                            info_to_bradley_terry=info_to_bradley_terry,
                            users_rank=users_rank,
                            category=category,
                            temperature=q3_temp,
                            doing_both_ways=args.do_both_ways
                        )
                    elif prompt_mode in Q1_THEN_Q2_PROMPTS:
                        log_or_print("-- Executing Q1 / Q2...", logger)

                        # ==============================================
                        # Q1
                        # ==============================================
                        if prompt_mode in Q1_THEN_Q2_PROMPTS:
                            proxann.do_q1(
                                prompter=prompter, 
                                cluster_data=cluster_data, 
                                users_cats=users_cats, 
                                categories=categories, 
                                temperature=q1_temp
                            )
                            category = categories[-1]
                        else:
                            category = None

                        # ==============================================
                        # Q2
                        # ==============================================
                        # if args.use_user_cats:
                        #    for_q2user_cats = users_cats
                        labels = proxann.do_q2(
                            prompter=prompter, 
                            prompt_mode=prompt_mode, 
                            cluster_data=cluster_data, 
                            fit_data=fit_data, 
                            category=category, 
                            temperature=q2_temp
                        )

                # llm loop ends here
                #  we save the results as if the LLMs are annotators
                llm_results_q1.append({
                    "id": id_,
                    "model": model,
                    "n_annotators": len(model_types),
                    "annotators": model_types,
                    "topic": cluster_id,
                    "topic_match_id": topic_match_id,
                    "categories": categories,
                    "user_categories": users_cats
                })

                if fit_data != []:
                    llm_results_q2.append({
                        "id": id_,
                        "model": model,
                        "n_annotators": len(model_types),
                        "annotators": model_types,
                        "topic": cluster_id,
                        "topic_match_id": topic_match_id,
                        "labels": labels,
                        "fit_data": [fit_data],
                    })

                if rank_data != []:
                    llm_results_q3.append({
                        "id": id_,
                        "model": model,
                        "n_annotators": len(model_types),
                        "annotators": model_types,
                        "topic": cluster_id,
                        "topic_match_id": topic_match_id,
                        "rank_data": rank_data
                    })
                
                if info_to_bradley_terry:
                    all_info_bradley_terry.append({
                        "id": id_,
                        "model": model,
                        "n_annotators": len(model_types),
                        "annotators": model_types,
                        "topic": cluster_id,
                        "topic_match_id": topic_match_id,
                        "info": info_to_bradley_terry,
                    })

    # prompt_mode loop ends here
    if llm_results_q2 == []:
        llm_results_q2 = None
    if llm_results_q3 == []:
        llm_results_q3 = None

    # Sort results by id
    if llm_results_q2 is not None:
        llm_results_q2 = sorted(llm_results_q2, key=lambda x: x["id"])
    if llm_results_q3 is not None:
        llm_results_q3 = sorted(llm_results_q3, key=lambda x: x["id"])
    if all_info_bradley_terry:
        all_info_bradley_terry = sorted(all_info_bradley_terry, key=lambda x: x["id"])

    ############################################################################
    # Save results
    ############################################################################
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    short_model = args.model_type.split("/")[-1]
    save_path = f"{args.path_save_results}/{short_model}/{args.prompt_mode}"
    if args.temperatures is not None:
        temp_str = "_".join(args.temperatures.split(","))
        save_path += f"_temp{temp_str}"
    if args.seed is not None:
        save_path += f"_seed{args.seed}"
    save_path += f"_{timestamp}"
    os.makedirs(save_path, exist_ok=True)
    save_results(llm_results_q1, save_path, "llm_results_q1.json")
    if llm_results_q2:
        save_results(llm_results_q2, save_path, "llm_results_q2.json")
    if llm_results_q3:
        save_results(llm_results_q3, save_path, "llm_results_q3.json")
    if all_info_bradley_terry:
        save_results(all_info_bradley_terry, save_path, "all_info_bradley_terry.json")

if __name__ == "__main__":
    main()
