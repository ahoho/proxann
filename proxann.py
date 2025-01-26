import argparse
from collections import Counter
import logging
import os
import datetime

import pandas as pd

from src.proxann.prompter import Prompter
from src.proxann.utils import (
    bradley_terry_model, collect_fit_rank_data, extract_info_binary_q2, extract_info_q1_q3, extract_logprobs, load_config_pilot, normalize_key, process_responses
)
from src.utils.utils import init_logger, load_yaml_config_file, log_or_print

Q1_THEN_Q2_PROMPTS = {"q1_then_binary_q2",
                      "q1_then_q2_fix_cat", "q1_then_q2_dspy"}
Q1_THEN_Q3_PROMPTS = {"q1_then_q3", "q1_then_q3_fix_cat", "q1_then_q3_dspy"}


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
        help="Run Q3 twice: once with A as the first document, then reversed.",
        default=False)
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

    assert os.path.exists(os.path.join(path, filename)
                          ), f"Error saving {filename}"


# =============================================================================
# Logic for Q1 / Q2 / Q3
# =============================================================================
def do_q1(
    prompter: Prompter,
    cluster_data: dict,
    users_cats: list,
    categories: list,
    dft_system_prompt: str = "src/proxann/prompts/q1/simplified_system_prompt.txt",
    logger: logging.Logger = None
) -> None:
    """Execute Q1.

    Parameters
    ----------
    prompter : Prompter
        Prompter object.
    cluster_data : Information for a topic as given by the data loaded from args.tm_model_data_path. 
    users_cats : List of user-generated categories during the user study.
    categories : List to store the LLM-generated categories.
    dft_system_prompt : str, optional
        Default system prompt for Q1, by default "src/proxann/prompts/q1/simplified_system_prompt.txt".
    logging : logging.Logger, optional
        Logger object, by default None
    """
    log_or_print("Executing Q1...", logger)

    question = prompter.get_prompt(cluster_data, "q1")
    category, _ = prompter.prompt(
        dft_system_prompt, question, use_context=False)  # max_tokens=10

    categories.append(category)
    log_or_print(f"\033[92mUser categories: {users_cats}\033[0m", logger)
    log_or_print(f"\033[94mModel category: {category}\033[0m", logger)

    return


def do_q2(
    prompter: Prompter,
    prompt_mode: str,
    llm_model: str,
    cluster_data: dict,
    fit_data: list,
    category: str,
    user_cats: list = None,
    dft_system_prompt: str = "src/proxann/prompts/XXX/simplified_system_prompt.txt",
    use_context: bool = False,
    logger: logging.Logger = None
) -> list:
    """
    Execute Q2.

    Parameters
    ----------
    prompter : Prompter
        Prompter object.
    prompt_mode : str
        Prompting mode for Q2.
    llm_model : str
        LLM model to use.
    cluster_data : dict
        Information for a topic as given by the data loaded from args.tm_model_data_path.
    fit_data : list
        List to store the fit scores.
    category : str
        LLM-generated category.
    user_cats : list, optional
        List of user-generated categories during the user study, by default None.
    dft_system_prompt : str, optional
        Default system prompt for Q2, by default "src/proxann/prompts/XXX/simplified_system_prompt.txt".
    use_context : bool, optional
        Whether to use the context for the prompt, by default False.

    Returns
    -------
    list
        List of user categories.
    """
    if prompt_mode == "q1_then_q2_dspy":
        if "llama" in llm_model:
            prompt_key = "q2_dspy_llama"
        elif "qwen" in llm_model:
            prompt_key = "q2_dspy_qwen"
        else:
            prompt_key = "q2_dspy"
        log_or_print(f"Using prompt key: {prompt_key}", logger)
        questions = prompter.get_prompt(cluster_data, prompt_key, category)
    else:
        do_q2_with_q1_fixed = prompt_mode == "q1_then_q2_fix_cat"
        questions = prompter.get_prompt(
            cluster_data, "binary_q2", category, do_q2_with_q1_fixed=do_q2_with_q1_fixed)

    if "dspy" in prompt_mode:
        dft_system_prompt = None

    if user_cats:
        labels = user_cats * len(questions)

        #  we do not to make one prompt per user category (each user has a different category), and we want to use each user's category to determine the fit score for each document in the evaluation set
        for cat in user_cats:
            if prompt_mode == "q1_then_q2_dspy":
                if "llama" in llm_model:
                    prompt_key = "q2_dspy_llama"
                elif "qwen" in llm_model:
                    prompt_key = "q2_dspy_qwen"
                else:
                    prompt_key = "q2_dspy"
                log_or_print(f"Using prompt key: {prompt_key}", logger)
                questions = prompter.get_prompt(cluster_data, prompt_key, cat)
            else:
                do_q2_with_q1_fixed = prompt_mode == "q1_then_q2_fix_cat"
                questions = prompter.get_prompt(
                    cluster_data, "binary_q2", cat, do_q2_with_q1_fixed=do_q2_with_q1_fixed)

            for question in questions:
                response_q2, _ = prompter.prompt(
                    dft_system_prompt, question, use_context=use_context)
                score = extract_info_binary_q2(response_q2)
                log_or_print(f"\033[92mFit: {score}\033[0m", logger)
                fit_data.append(score)
    else:
        labels = [category] * len(questions)
    for question in questions:
        response_q2, _ = prompter.prompt(
            dft_system_prompt, question, use_context=use_context)
        log_or_print(f"\033[96mFit: {response_q2}\033[0m", logger)
        score = extract_info_binary_q2(response_q2)
        #if "marginally" in response_q2.lower() or "marginal" in response_q2.lower() or "maybe" in response_q2.lower():
        #if "no" in response_q2.lower():
        #    import pdb; pdb.set_trace()
        log_or_print(f"\033[92mFit: {score}\033[0m", logger)
        fit_data.append(score)

    return labels


def do_q3(
    prompter: Prompter,
    prompt_mode: str,
    llm_model: str,
    cluster_data: dict,
    rank_data: list,
    users_rank: list,
    category: str,
    doing_both_ways: bool = False,
    dft_system_prompt: str = "src/proxann/prompts/q3/simplified_system_prompt.txt",
    use_context: bool = False,
    logger: logging.Logger = None
) -> None:
    """
    Execute Q3.

    Parameters
    ----------
    prompter : Prompter
        Prompter object.
    prompt_mode : str
        Prompting mode for Q3.
    llm_model : str
        LLM model to use.
    cluster_data : dict
        Information for a topic as given by the data loaded from args.tm_model_data_path.
    rank_data : list
        List to store the rank data.
    users_rank : list
        List of user ranks.
    category : str
        LLM-generated category.
    doing_both_ways : bool, optional
        Whether to run Q3 twice: once with A as the first document, then reversed, by default False.
    dft_system_prompt : str, optional
        Default system prompt for Q3, by default "src/proxann/prompts/q3/simplified_system_prompt.txt".
    use_context : bool, optional
        Whether to use the context for the prompt, by default False.
    logger : logging.Logger, optional
        Logger object, by default None.
    """
    # if "dspy" in prompt_mode:
    #   dft_system_prompt = None

    do_q3_with_q1_fixed = prompt_mode == "q1_then_q3_fix_cat"

    if prompt_mode == "q1_then_q3_dspy":
        prompt_key = "q3_dspy_llama" if "llama" in llm_model else "q3_dspy"
        log_or_print(f"-- Using prompt key: {prompt_key}", logger)
    else:
        prompt_key = "q3"
    q3_out = prompter.get_prompt(cluster_data, prompt_key, category=category, do_q3_with_q1_fixed=do_q3_with_q1_fixed, doing_both_ways=doing_both_ways)

    if isinstance(q3_out, tuple) and len(q3_out) > 2:  # Both ways
        questions_one, pair_ids_one, questions_two, pair_ids_two = q3_out
        ways = [[questions_one, pair_ids_one], [questions_two, pair_ids_two]]
    else:  # Single way
        questions, pair_ids = q3_out
        ways = [[questions, pair_ids]]

    labels_one, orders_one, rationales_one, logprobs_one = [], [], [], []
    labels_two, orders_two, rationales_two, logprobs_two = [], [], [], []

    for way_id, (questions, pair_ids) in enumerate(ways):
        log_or_print(
            f"-- Executing Q3 ({'both ways' if len(ways) > 1 else 'one way'})...", logger)
        for question in questions:
            pairwise, pairwise_logprobs = prompter.prompt(
                dft_system_prompt, question, use_context=use_context
            )
            try:
                label, order, rationale = extract_info_q1_q3(
                    pairwise, get_label=(prompt_mode == "q1_and_q3"))
                if len(ways) > 1 and way_id == 0:
                    labels_one.append(label)
                    orders_one.append(order)
                    rationales_one.append(rationale)
                    logprobs_one.append(extract_logprobs(
                        pairwise_logprobs, prompter.backend, logger))
                    log_or_print(f"\033[92mOrder: {order}\033[0m", logger)
                elif len(ways) > 1 and way_id == 1:
                    labels_two.append(label)
                    orders_two.append(order)
                    rationales_two.append(rationale)
                    logprobs_two.append(extract_logprobs(
                        pairwise_logprobs, prompter.backend, logger))
                    log_or_print(f"\033[94mOrder: {order}\033[0m", logger)
                else:
                    labels_one.append(label)
                    orders_one.append(order)
                    rationales_one.append(rationale)
                    logprobs_one.append(extract_logprobs(
                        pairwise_logprobs, prompter.backend, logger))
                    log_or_print(f"\033[92mOrder: {order}\033[0m", logger)
            except Exception as e:
                log_or_print(
                    f"-- Error extracting info from prompt: {e}", "error", logger)

    # Combine results for ranking
    if len(ways) > 1:
        pair_ids_comb = ways[0][1] + ways[1][1]
        orders_comb = orders_one + orders_two
        logprobs_comb = logprobs_one + logprobs_two
    else:
        pair_ids_comb = ways[0][1]
        orders_comb = orders_one
        logprobs_comb = logprobs_one

    # Rank computation
    ranked_documents = bradley_terry_model(
        pair_ids_comb, orders_comb, logprobs_comb)
    true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
    ranking_indices = {doc_id: idx for idx, doc_id in enumerate(ranked_documents['doc_id'])}
    rank = [ranking_indices[doc_id] + 1 for doc_id in true_order]
    rank = [len(rank) - r + 1 for r in rank]  # Invert rank

    log_or_print(f"\033[95mLLM Rank:\n {rank}\033[0m", logger)
    log_or_print(f"\033[95mUsers rank: {users_rank}\033[0m", logger)
    rank_data.append(rank)

    return


def main():
    args = parse_args()

    # Init logger and load config
    logger = init_logger(args.config_path, f"RunProxann-{args.running_mode}")
    logger.info(f"Running Proxann in mode: {args.running_mode}")
    config = load_yaml_config_file(args.config_path, "user_study", logger)

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

    llm_results_q1, llm_results_q2, llm_results_q3 = [], [], []
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
                users_cats = [user_data["category"]
                              for user_data in responses_by_id[id_]]
                this_corr_data = next(c for c in corr_data if c["id"] == id_)
                users_rank = this_corr_data["rank_data"]

                # ---------------------------------------------------------
                # For each LLM ...
                # ---------------------------------------------------------
                # to store the rank data for each lmm
                rank_data = []  # it will store the rank data for each lmm
                fit_data = []  # it will store the fit data for each lmm
                categories = []  # it will store the categories for each lmm

                for llm_model in model_types:
                    log_or_print(f"LLM: {llm_model}", logger)
                    prompter = Prompter(model_type=llm_model)

                    if prompt_mode in Q1_THEN_Q3_PROMPTS:
                        log_or_print("-- Executing Q1 / Q3...", logger)

                        # ==============================================
                        # Q1
                        # ==============================================
                        if prompt_mode in Q1_THEN_Q3_PROMPTS:
                            do_q1(prompter, cluster_data, users_cats, categories)
                            category = categories[-1]
                        else:
                            category = None

                        # ==============================================
                        # Q3
                        # ==============================================
                        # TODO: Add logic for when category is not category[-1] but each of the users' categories
                        do_q3(prompter, prompt_mode, llm_model, cluster_data,rank_data, users_rank, category, args.do_both_ways)

                    elif prompt_mode in Q1_THEN_Q2_PROMPTS:
                        log_or_print("-- Executing Q1 / Q2...", logger)

                        # ==============================================
                        # Q1
                        # ==============================================
                        if prompt_mode in Q1_THEN_Q2_PROMPTS:
                            do_q1(prompter, cluster_data, users_cats, categories)
                            category = categories[-1]
                        else:
                            category = None

                        # ==============================================
                        # Q2
                        # ==============================================
                        # if args.use_user_cats:
                        #    for_q2user_cats = users_cats
                        labels = do_q2(prompter, prompt_mode, llm_model, cluster_data, fit_data, category)

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

    ############################################################################
    # Save results
    ############################################################################
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{args.path_save_results}/{args.prompt_mode}_{args.model_type}_{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    save_results(llm_results_q1, save_path, "llm_results_q1.json")
    if llm_results_q2:
        save_results(llm_results_q2, save_path, "llm_results_q2.json")
    if llm_results_q3:
        save_results(llm_results_q3, save_path, "llm_results_q3.json")


if __name__ == "__main__":
    main()
