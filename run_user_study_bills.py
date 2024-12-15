import argparse
import logging
import os
import time
import datetime
import re
from dotenv import load_dotenv
import dspy
import pandas as pd
from collections import Counter
from scipy.stats import kendalltau
from src.llm_eval.optimize_dspy import Q3Module
import numpy as np
from src.llm_eval.prompter import Prompter
from src.llm_eval.utils import (
    bradley_terry_model, calculate_corr_npmi, collect_fit_rank_data, compute_agreement_per_topic, 
    compute_correlations_one, compute_correlations_two, extract_info_binary_q2, 
    extract_info_q1_q3, extract_logprobs, load_config_pilot, 
    process_responses, calculate_coherence, normalize_key
)

Q1_THEN_Q2_PROMPTS = {"q1_then_binary_q2", "q1_then_q2_fix_cat", "q1_then_q2_dspy"}
Q1_THEN_Q3_PROMPTS = {"q1_then_q3", "q1_then_q3_fix_cat", "q1_then_q3_dspy"}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama3.2", help="Model types to evaluate")
    parser.add_argument("--prompt_mode", type=str, default="q1_then_q3", help="Prompting mode")
    parser.add_argument("--config_path", type=str, help="Path to config JSON files (comma-separated)", default="data/files_pilot/config_first_round.json,data/files_pilot/config_second_round.json")
    parser.add_argument("--response_csv", type=str, help="Path to responses CSV file", default="data/files_pilot/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv")
    parser.add_argument("--npmi_save", type=str, help="Path to save NPMI data", default="data/files_pilot/npmi.csv")
    parser.add_argument("--do_both_ways", action="store_true", help="Do both ways for Q3", default=False)
    parser.add_argument("use_user_cats", action="store_true", help="Use user categories for Q2/Q3", default=False)
    parser.add_argument("--removal_condition", type=str, help="Removal condition for responses", default="loose")
    parser.add_argument("--path_save_results", type=str, help="Path to save results", default="data/files_pilot/results")
    return parser.parse_args()

def save_results(data, path, filename):
    # check if data exists and if it is a DataFrame
    if data is None or len(data) == 0:
        print(f"-- -- No data to save for {filename}")
        return
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    data.to_json(os.path.join(path, filename), orient="records")
    
    assert os.path.exists(os.path.join(path, filename)), f"Error saving {filename}"

def load_or_calculate_npmi(config_pilot, npmi_save):
    if os.path.exists(npmi_save):
        logging.info("Loading existing NPMI data...")
        return pd.read_csv(npmi_save)
    logging.info("Calculating NPMI data...")
    start_time = time.time()
    npmi_data = calculate_coherence(config_pilot)
    npmi_data.to_csv(npmi_save, index=False)
    logging.info(f"NPMI calculated in {(time.time() - start_time) / 60:.2f} minutes.")
    return npmi_data

def do_q1(prompter, cluster_data, users_cats, categories, dft_system_prompt="src/llm_eval/prompts/q1/simplified_system_prompt.txt"):
    logging.info("-- Executing Q1...")
    
    question = prompter.get_prompt(cluster_data, "q1_bills")
    category, _ = prompter.prompt(dft_system_prompt, question, use_context=False) #max_tokens=10
    
    #pattern = r"\[\[ ## CATEGORY ## \]\]\s*(.+)"
    #match = re.search(pattern, category)
    #category = match.group(1) if match else category
    
    categories.append(category)
    print(f"\033[92mUser categories: {users_cats}\033[0m")
    print(f"\033[94mModel category: {category}\033[0m")
    
    return

def do_q2(prompter, prompt_mode, llm_model, cluster_data, fit_data, category, user_cats=None, dft_system_prompt="src/llm_eval/prompts/XXX/simplified_system_prompt.txt", use_context=False):
        
    if prompt_mode == "q1_then_q2_dspy":
        prompt_key = "q2_dspy" if "gpt" in llm_model else "q2_dspy_llama"
        logging.info(f"-- Using prompt key: {prompt_key}")
        questions = prompter.get_prompt(cluster_data, prompt_key, category)
    else:    
        do_q2_with_q1_fixed = prompt_mode == "q1_then_q2_fix_cat"
        questions = prompter.get_prompt(cluster_data, "binary_q2", category, do_q2_with_q1_fixed=do_q2_with_q1_fixed)
    
    if "dspy" in prompt_mode:
        dft_system_prompt = None
    #else:
    #    dft_system_prompt = dft_system_prompt.replace("XXX", "binary_q2")
    
    if user_cats:
        labels = user_cats * len(questions)
        
        # we do not to make one prompt per user category (each user has a different category), and we want to use each user's category to determine the fit score for each document in the evaluation set
        for cat in user_cats:
            if prompt_mode == "q1_then_q2_dspy":
                prompt_key = "q2_dspy" if "gpt" in llm_model else "q2_dspy_llama"
                logging.info(f"-- Using prompt key: {prompt_key}")
                questions = prompter.get_prompt(cluster_data, prompt_key, cat)
            else:    
                do_q2_with_q1_fixed = prompt_mode == "q1_then_q2_fix_cat"
                questions = prompter.get_prompt(cluster_data, "binary_q2", cat, do_q2_with_q1_fixed=do_q2_with_q1_fixed)
            
            for question in questions:
                response_q2, _ = prompter.prompt(dft_system_prompt, question, use_context=use_context)
                score = extract_info_binary_q2(response_q2)
                print(f"\033[92mFit: {score}\033[0m")
                fit_data.append(score)
    else:
        labels = [category] * len(questions)
    for question in questions:
        response_q2, _ = prompter.prompt(dft_system_prompt, question, use_context=use_context)
        score = extract_info_binary_q2(response_q2)
        print(f"\033[92mFit: {score}\033[0m")
        fit_data.append(score)
    
    return labels
    
def do_q3(prompter, prompt_mode, llm_model, cluster_data, rank_data, users_rank, category, doing_both_ways, dft_system_prompt="src/llm_eval/prompts/q3/simplified_system_prompt.txt", use_context=False):
    
    #if "dspy" in prompt_mode:
    #   dft_system_prompt = None
    
    do_q3_with_q1_fixed = prompt_mode == "q1_then_q3_fix_cat"

    if prompt_mode == "q1_then_q3_dspy":
        prompt_key = "q3_dspy" if "gpt" in llm_model else "q3_dspy_llama"
        logging.info(f"-- Using prompt key: {prompt_key}")
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
        logging.info(f"-- Executing Q3 ({'both ways' if len(ways) > 1 else 'one way'})...")
        for question in questions:
            pairwise, pairwise_logprobs = prompter.prompt(
                dft_system_prompt, question, use_context=use_context
            )
            try:
                label, order, rationale = extract_info_q1_q3(pairwise, get_label=(prompt_mode == "q1_and_q3"))
                if len(ways) > 1 and way_id == 0:
                    labels_one.append(label)
                    orders_one.append(order)
                    rationales_one.append(rationale)
                    logprobs_one.append(extract_logprobs(pairwise_logprobs, prompter.backend, logging))
                    print(f"\033[92mOrder: {order}\033[0m")
                elif len(ways) > 1 and way_id == 1:
                    labels_two.append(label)
                    orders_two.append(order)
                    rationales_two.append(rationale)
                    logprobs_two.append(extract_logprobs(pairwise_logprobs, prompter.backend, logging))
                    print(f"\033[94mOrder: {order}\033[0m")
                else:
                    labels_one.append(label)
                    orders_one.append(order)
                    rationales_one.append(rationale)
                    logprobs_one.append(extract_logprobs(pairwise_logprobs, prompter.backend, logging))
                    print(f"\033[92mOrder: {order}\033[0m")
            except Exception as e:
                logging.error(f"-- Error extracting info from prompt: {e}")

    # Combine results for ranking
    if len(ways) > 1:
        pair_ids_comb = ways[0][1] + ways[1][1]
        orders_comb = orders_one + orders_two
        logprobs_comb = logprobs_one + logprobs_two
    else:
        pair_ids_comb = ways[0][1]
        orders_comb = orders_one
        logprobs_comb = logprobs_one
    
    
    """
    #lm = dspy.LM(
    #    "ollama_chat/llama3.1:8b-instruct-q8_0",
    #    api_base="http://kumo01:11434"
    #
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    lm = dspy.LM(model="gpt-4o-mini-2024-07-18")
    dspy.settings.configure(lm=lm)
    
    q3_dspy_prompter = Q3Module()
    q3_dspy_prompter.load("/Users/lbartolome/Documents/GitHub/theta-evaluation/data/dspy-saved/gpt-4o-mini-2024-07-18_v2_10dec.json")
    orders_comb = []
    logprobs_comb = None
    cat, pairs, pair_ids_comb = q3_out
    for pair in pairs:
        order = q3_dspy_prompter(cat, pair[0], pair[1]).closest
        print(f"ORDER: {order}")
        orders_comb.append(order)
    """

    # Rank computation
    ranked_documents = bradley_terry_model(pair_ids_comb, orders_comb, logprobs_comb)
    true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
    ranking_indices = {doc_id: idx for idx, doc_id in enumerate(ranked_documents['doc_id'])}
    rank = [ranking_indices[doc_id] + 1 for doc_id in true_order]
    rank = [len(rank) - r + 1 for r in rank]  # Invert rank

    print(f"\033[95mLLM Rank:\n {rank}\033[0m")
    print(f"\033[95mUsers rank: {users_rank}\033[0m")
    rank_data.append(rank)
        
    return

def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # TODO: Pass this as an argument or define somewhere else
    valid_models={"mallet", "ctm", "bertopic", "category-45"}
    valid_datasets={"wikitext-labeled", "bills-labeled"}

    config_pilot = load_config_pilot(args.config_path)
    # normalize keys in config_pilot
    config_pilot = {normalize_key(key, valid_models, valid_datasets, False): value for key, value in config_pilot.items()}
    
    responses_by_id = process_responses(args.response_csv, args.config_path.split(","), removal_condition = args.removal_condition)
    _, _, _, corr_data = collect_fit_rank_data(responses_by_id)
        
    #res = compute_agreement_per_topic(responses_by_id)
        
    # Load or calculate NPMI data
    #npmi_data = load_or_calculate_npmi(config_pilot, args.npmi_save)
    
    
    model_types = args.model_type.split(",") if args.model_type else []
    prompt_modes = args.prompt_mode.split(",") if args.prompt_mode else []
    
    llm_results_q1, llm_results_q2, llm_results_q3 = [], [], []
    topics_per_model = Counter()
    
    for prompt_mode in prompt_modes:
        logging.info(f"-- -- Executing in MODE: {prompt_mode} -- --")
        
        # ---------------------------------------------------------
        # For each topic model (mallet / ctm / bertopic) ... 
        # ---------------------------------------------------------
        # this selects the set of topics for a given model (mallet, ctm, etc.)
        for model_id, model_data in config_pilot.items(): 
            logging.info(f"-- -- Model: {model_id}")
            
            # ---------------------------------------------------------
            # For each topic ... 
            # ---------------------------------------------------------
            # each cluster_data is the information for a topic
            for cluster_id, cluster_data in model_data.items(): 
                logging.info(f"-- -- Cluster: {cluster_id}")
                
                # topic information
                id_ = f"{model_id}/{cluster_id}"
                model = model_id.split("/")[-1]
                topics_per_model[model_id] += 1
                topic_match_id = topics_per_model[model_id]
                
                # user data (categories, ranks)
                #users_cats = []
                #users_rank = []
                users_cats = [user_data["category"] for user_data in responses_by_id[id_]]
                this_corr_data = next(c for c in corr_data if c["id"] == id_)
                users_rank = this_corr_data["rank_data"]
                
                # ---------------------------------------------------------
                # For each LLM ... 
                # ---------------------------------------------------------
                # to store the rank data for each lmm
                rank_data = [] # it will store the rank data for each lmm 
                fit_data = [] # it will store the fit data for each lmm
                categories = [] # it will store the categories for each lmm
                
                for llm_model in model_types:
                    logging.info(f"-- -- -- -- LLM: {llm_model}")
                    prompter = Prompter(model_type=llm_model)
                    
                    if prompt_mode in Q1_THEN_Q3_PROMPTS:
                        logging.info("-- Executing Q1 / Q3...")
                        
                        #==============================================
                        # Q1
                        #==============================================
                        if prompt_mode in Q1_THEN_Q3_PROMPTS:
                            do_q1(prompter, cluster_data, users_cats, categories)
                            category = categories[-1]
                        else:
                            category = None
                        
                        #==============================================
                        # Q3
                        #==============================================
                        # TODO: Add logic for when category is not category[-1] but each of the users' categories
                        do_q3(prompter, prompt_mode, llm_model, cluster_data, rank_data, users_rank, category, args.do_both_ways)
                    
                    elif prompt_mode in Q1_THEN_Q2_PROMPTS:
                        logging.info("-- Executing Q1 / Q2...")
                        
                        #==============================================
                        # Q1
                        #==============================================
                        if prompt_mode in Q1_THEN_Q2_PROMPTS:
                            do_q1(prompter, cluster_data, users_cats, categories)
                            category = categories[-1]
                        else:
                            category = None
                            
                        #==============================================
                        # Q2
                        #==============================================
                        #if args.use_user_cats:
                        #    for_q2user_cats = users_cats
                        labels = do_q2(prompter, prompt_mode, llm_model, cluster_data, fit_data, category)
                            
                # llm loop ends here
                # we save the results as if the LLMs are annotators
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
    #npmi_data = npmi_data.sort_values(by=["id"])
    corr_data = sorted(corr_data, key=lambda x: x["id"])
        
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
        
    #import pdb; pdb.set_trace()
            
    # Correlations with user study data and ground truth
    corr_results = compute_correlations_one(corr_data,rank_llm_data=llm_results_q3, fit_llm_data=llm_results_q2)
    save_results(corr_results, save_path, "correlation_results_mode1.json")
    
    # aux = pd.DataFrame(corr_data)
    # aux["fit_bin"] = aux["fit_data"].apply(lambda x: (np.round(np.mean(x, axis=0)) >= 4).astype(int))
    
    # corr_results['rank_tau_users_llama3.1:8b-instruct-q8_0'].mean()
    # kendalltau(corr_results["rank_tau"], corr_results["rank_tau_tm_llama3.1:8b-instruct-q8_0"])
    
    # check tau and significance
    if llm_results_q3 is not None:
        for llm_model in model_types:
            kendall_tau_tau_tm = kendalltau(corr_results["rank_tau"], corr_results[f"rank_tau_tm_{llm_model}"])
            # save kendalltau
            kendall_tau_tau_tm = pd.DataFrame([{"tau": kendall_tau_tau_tm.correlation, "p_value": kendall_tau_tau_tm.pvalue}])
            save_results(kendall_tau_tau_tm, save_path, f"kendall_tau_tau_tm_{llm_model}.json")
            print(f"Kendall Tau {llm_model}: {kendall_tau_tau_tm}")
    
    corr_results2 = compute_correlations_two(responses_by_id,llm_results_q3, llm_results_q2)
    save_results(corr_results2, save_path, "correlation_results_mode2.json")
    # NPMI
    columns_calculate = [col for col in corr_results.columns if col not in["id", "model", "topic", "n_annotators", "fit_rho", "fit_tau", "fit_agree"] and "rank_rho_users" not in col and "rank_tau_users" not in col]
    #npmi_corr_results = calculate_corr_npmi(corr_results, npmi_data, columns_calculate)
    #save_results(npmi_corr_results, save_path, "npmi_corr_results.json")
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()