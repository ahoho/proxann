import argparse
from collections import Counter
import logging
import datetime
import os
import time
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from src.llm_eval.prompter import Prompter
from src.llm_eval.utils import bradley_terry_model, calculate_corr_npmi, collect_fit_rank_data, compute_agreement_per_topic, compute_correlations_one, compute_correlations_two, extract_info_q1_q3, extract_logprobs, load_config_pilot, process_responses, extract_info_q1_q2, generate_pairwise_counts, calculate_coherence

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama3.2")
    parser.add_argument("--prompt_mode", type=str, default="q1_then_q3")
    parser.add_argument("--config_path", type=str, default="data/files_pilot/config_first_round.json,data/files_pilot/config_second_round.json")
    parser.add_argument("--response_csv", type=str, default="data/files_pilot/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv")
    return parser.parse_args()

def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Read data from user study
    config_pilot = load_config_pilot(args.config_path)
    
    # Read results from user study
    responses_by_id = process_responses(args.response_csv, args.config_path.split(","))
    _, _, _, corr_data = collect_fit_rank_data(responses_by_id)
    
    # Calculate coherence
    npmi_save = "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/files_pilot/npmi.csv"
    if os.path.exists(npmi_save):
        logging.info("-- Loading NPMI data --")
        npmi_data = pd.read_csv(npmi_save)
    else:
        logging.info("-- Calculating NPMI data --")
        time_start = time.time()
        npmi_data = calculate_coherence(config_pilot)
        npmi_data.to_csv(npmi_save, index=False)
        time_npmi = (time.time() - time_start) / 60
        logging.info(f"-- NPMI data calculated in {time_npmi} minutes--")

    model_types = args.model_type.split(",") if args.model_type else []
    prompt_modes = args.prompt_mode.split(",") if args.prompt_mode else []
    
    llm_results_q1, llm_results_q2, llm_results_q3 = [], [], []
    topics_per_model = Counter()
    
    # keep dictionary with only first key (for debugging)
    """
    inside_dict = {
        '15':  config_pilot['data/models/ctm']['15']
    }
    config_pilot = {
        'data/models/ctm': inside_dict
    }
    """
    for prompt_mode in prompt_modes:
        logging.info(f"-- -- Executing in MODE: {prompt_mode} -- --")
        for model_id, model_data in config_pilot.items(): # this selects the set of topics for a given model (mallet, ctm, etc.)
            logging.info(f"-- -- Model: {model_id}")
            for cluster_id, cluster_data in model_data.items(): # each cluster_data is the information for a topic
                logging.info(f"-- -- Cluster: {cluster_id}")
                id_ = f"{model_id}/{cluster_id}"
                model = model_id.split("/")[-1]
                topics_per_model[model_id] += 1
                topic_match_id = topics_per_model[model_id]
                
                users_cats = [user_data["category"] for user_data in responses_by_id[id_]]
                
                true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
                
                this_corr_data = [c for c in corr_data if c["id"] == id_][0]
                
                # get "ground truth" rank
                users_rank = this_corr_data["rank_data"]
                # rank data is an array of arrays, each array is the rank of the documents for a user
                # we need to convert each row as [len(row) - r + 1 for r in row]
                users_rank_rev = np.array([[len(row) - r + 1 for r in row] for row in users_rank])
                gt = generate_pairwise_counts(users_rank)
                gt_sums = gt.sum(axis=1)
                rank_to_doc_id = {doc_id: rank for doc_id, rank in zip(true_order, users_rank_rev.T)}
                
                # get "ground truth" fit
                users_fit = this_corr_data["fit_data"].T
                fit_to_doc_id = {doc_id: fit for doc_id, fit in zip(true_order, users_fit)}
                
                # to store the rank data for each lmm
                rank_data = [] # it will store the rank data for each lmm 
                fit_data = [] # it will store the fit data for each lmm
                categories = [] # it will store the categories for each lmm
                
                for llm_model in model_types:
                    logging.info(f"-- -- -- -- LLM: {llm_model}")
                    prompter = Prompter(model_type=llm_model)
                    
                    if prompt_mode == "q1_then_q3" or prompt_mode == "q1_and_q3" or prompt_mode == "q1_then_q3_fix_cat":

                        if prompt_mode == "q1_then_q3" or prompt_mode == "q1_then_q3_fix_cat":
                            
                            #==============================================
                            # Q1
                            #==============================================
                            logging.info("-- Executing Q1...")
                            
                            question = prompter.get_prompt(cluster_data, "q1")
                            category, _ = prompter.prompt("src/llm_eval/prompts/q1/simplified_system_prompt.txt", question, use_context=False)
                            categories.append(category)
                            print(f"\033[92mUser categories: {users_cats}\033[0m")
                            print(f"\033[94mModel category: {category}\033[0m")
                            
                            #==============================================
                            # Q3
                            #==============================================
                            logging.info("-- Executing Q3...")
                            do_q3_with_q1_fixed = prompt_mode == "q1_then_q3_fix_cat"
                            q3_out = prompter.get_prompt(cluster_data, "q3", category, do_q3_with_q1_fixed=do_q3_with_q1_fixed)
                            get_label = False
                            
                            if len(q3_out) > 2: ## then is "doing it both ways"
                                questions_one, pair_ids_one, questions_two, pair_ids_two = q3_out
                                questions = None
                            else:
                                questions, pair_ids = q3_out
                                questions_one = None
                                
                            # questions, pair_ids
                        elif prompt_mode == "q1_and_q3":
                            questions, pair_ids = prompter.get_prompt(cluster_data, "q1_q3")
                            get_label = True

                        if questions is None: # we need to save double the data
                            labels_one, orders_one, rationales_one, logprobs_one = [], [], [], []
                            labels_two, orders_two, rationales_two, logprobs_two = [], [], [], []
                        else:
                            labels, orders, rationales, logprobs = [], [], [], []
                        
                        if questions is None: # then it is "doing it both ways"
                            ways = [[questions_one,pair_ids_one], [questions_two, pair_ids_two]]
                        else: 
                            ways = [[questions, pair_ids]]

                        num_questions = len(ways[0][0])
                        for id_q in range(num_questions): 
                            for way_id, way in enumerate(ways):
                                questions, pair_ids = way
                                question = questions[id_q]  
                                if len(ways) > 1:
                                    logging.info("-- Executing Q3 (both ways)...")
                                else:
                                    logging.info("-- Executing Q3 (one way)...")

                                # Run prompt and process the response
                                pairwise, pairwise_logprobs = prompter.prompt(
                                    "src/llm_eval/prompts/q1_q3/simplified_system_prompt.txt" if prompt_mode == "q1_and_q3" else "src/llm_eval/prompts/q3/simplified_system_prompt.txt",
                                    question,
                                    use_context=False
                                )

                                try:
                                    label, order, rationale = extract_info_q1_q3(pairwise, get_label=get_label)
                                    if order == "":
                                        pairwise, pairwise_logprobs = prompter.prompt(
                                            "src/llm_eval/prompts/q1_q3/simplified_system_prompt.txt" if prompt_mode == "q1_and_q3" else "src/llm_eval/prompts/q3/simplified_system_prompt.txt",
                                            question,
                                            use_context=False
                                        )
                                        label, order, rationale = extract_info_q1_q3(pairwise, get_label=get_label)
                                        if order == "":
                                            logging.warning(f"-- -- No order extracted for model {model_id} and cluster {cluster_id}")
                                            import pdb; pdb.set_trace()

                                    if len(ways) > 1 and way_id == 0:
                                        labels_one.append(label)
                                        orders_one.append(order)
                                        rationales_one.append(rationale)
                                        print(f"\033[92mOrder: {order}\033[0m")
                                    elif len(ways) > 1 and way_id == 1:
                                        labels_two.append(label)
                                        orders_two.append(order)
                                        rationales_two.append(rationale)
                                        print(f"\033[94mOrder: {order}\033[0m")
                                    else:
                                        print(f"\033[92mOrder: {order}\033[0m")
                                        labels.append(label)
                                        orders.append(order)
                                        rationales.append(rationale)
                                except Exception as e:
                                    logging.error(f"-- -- Error extracting info from prompt: {e}")

                                # Extract logprobs if available
                                if pairwise_logprobs is not None:
                                    prob_values = extract_logprobs(pairwise_logprobs, prompter.backend, logging)
                                    if prob_values:
                                        if len(ways) > 1 and way_id == 0:
                                            logprobs_one.append(prob_values[0])
                                        elif len(ways) > 1 and way_id == 1:
                                            logprobs_two.append(prob_values[0])
                                        else:
                                            logprobs.append(prob_values[0])
                                    else:
                                        logging.warning(f"-- -- No logprobs extracted for model {model_id} and cluster {cluster_id}")

                            # check t-tests
                            # fit_to_doc_id
                            # pair_ids[id_q]
                            t_test_fit = ttest_ind(fit_to_doc_id[pair_ids[id_q]["A"]], fit_to_doc_id[pair_ids[id_q]["B"]])
                            t_test_rank = ttest_ind(rank_to_doc_id[pair_ids[id_q]["A"]], rank_to_doc_id[pair_ids[id_q]["B"]])
                            
                            pvalue_thr = 0.05
                            if t_test_fit.pvalue < pvalue_thr or t_test_rank.pvalue < pvalue_thr:
                                output_lines = []
                                output_lines.append(f"- LLM: {llm_model}")
                                output_lines.append(f"- Model: {model_id} - Cluster: {cluster_id}")
                                output_lines.append(f"* CATEGORY: {category}")
                                output_lines.append(f"* USER CATEGORIES: {users_cats}")

                                for test_name, test in [("Fit", t_test_fit), ("Rank", t_test_rank)]:
                                    significance = "is" if test.pvalue < pvalue_thr else "is not"
                                    output_lines.append(f"-- -- {test_name} t-test {significance} statistically significant at {pvalue_thr}: {test}")

                                closest_one = pair_ids_one[id_q]["A"] if orders_one[-1] == "A" else pair_ids_one[id_q]["B"]
                                closest_two = pair_ids_two[id_q]["A"] if orders_two[-1] == "A" else pair_ids_two[id_q]["B"]

                                output_lines.append(f"-- -- CLOSEST 'one-way' (pair ids: {pair_ids_one[id_q]}): {closest_one}")
                                output_lines.append(f"-- -- CLOSEST 'other-way' (pair ids: {pair_ids_two[id_q]}): {closest_two}")

                                best_fit = pair_ids[id_q]['A'] if t_test_fit.statistic > 0 else pair_ids[id_q]['B']
                                best_rank = pair_ids[id_q]['A'] if t_test_rank.statistic > 0 else pair_ids[id_q]['B']

                                for test_name, best in [("fit scores", best_fit), ("rank", best_rank)]:
                                    output_lines.append(f"-- -- Based on {test_name}, users say that {best} is better")

                                # disagreements and save outputs to file
                                if closest_one != closest_two:
                                    output_lines.append("-- -- Disagreement found: LLMs closest documents in each way are different")
                                    output_lines.append(f"-- -- RATIONALE ONE: {rationales_one[-1]}")
                                    output_lines.append(f"-- -- RATIONALE TWO: {rationales_two[-1]}")
                                    
                                else:
                                    if closest_one != best_fit:
                                        output_lines.append("-- -- Disagreement found: The users fit does not agree with the best one")
                                        output_lines.append(rationales_one[-1])
                                    else:
                                        output_lines.append("-- -- The users fit agrees with the best one")
                                    if closest_one != best_rank:
                                        output_lines.append("-- -- Disagreement found: The users rank does not agree with the best one")
                                        output_lines.append(rationales_one[-1])
                                    else:
                                        output_lines.append("-- -- The users rank agrees with the best one")
                                    if best_fit != best_rank:
                                        output_lines.append("-- -- Disagreement found: The users fit does not agree with the users rank")
                                    else:
                                        output_lines.append("-- -- The users fit agrees with the users rank")                                    
                                #import pdb; pdb.set_trace()
                                with open('all_topics_logs.txt', 'a') as f:
                                    f.write('\n'.join(output_lines) + '\n\n')
                                    
                                    #import pdb; pdb.set_trace()

                                # Print all outputs
                                for line in output_lines:
                                    print(line)

                        if len(ways) > 1:
                            if (not logprobs_one or not logprobs_two):
                                logprobs_one = [None] * len(orders_one)
                                logprobs_two = [None] * len(orders_two)
                                logging.warning(f"-- -- No logprobs found for model {model_id} and cluster {cluster_id}")
                        else:
                            if not logprobs:
                                logprobs = [None] * len(orders)
                                logging.warning(f"-- -- No logprobs found for model {model_id} and cluster {cluster_id}")
                                
                        # print orders of each way in a different color
                        print(f"\033[92mOrders: {orders_one}\033[0m")
                        print(f"\033[94mOrders: {orders_two}\033[0m")

                        pair_ids_comb = pair_ids_one + pair_ids_two if len(ways) > 1 else pair_ids
                        orders_comb = orders_one + orders_two if len(ways) > 1 else orders
                        logprobs_comb = logprobs_one + logprobs_two if len(ways) > 1 else logprobs
                                                
                        # Obtain full rank (Bradley-Terry model)
                        ranked_documents = bradley_terry_model(pair_ids_comb, orders_comb, logprobs_comb)
                        
                        true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
                        
                        ranking_indices = {doc_id: idx for idx, doc_id in enumerate(ranked_documents['doc_id'])}
                        rank = [ranking_indices[doc_id] +1 for doc_id in true_order]    
                        rank = [len(rank) - r + 1 for r in rank] # Invert rank
                        
                        rank_data.append(rank)
                        
                        print(f"\033[95mLLM Rank: {rank}\033[0m")
                        print(f"\033[95mUsers rank: {users_rank}\033[1m")
                        print(f"\033[95mGT sum: {gt_sums}\033[1m")
                        
                        #import pdb; pdb.set_trace()                   
                    elif prompt_mode == "q1_then_q2" or prompt_mode == "q1_and_q2":
                        
                        if prompt_mode == "q1_then_q2":
                            logging.info("-- Executing Q1...")
                            question = prompter.get_prompt(cluster_data, "q1")
                            category, _ = prompter.prompt("src/llm_eval/prompts/q1/simplified_system_prompt.txt", question, use_context=False)
                            
                            logging.info("-- Executing Q2...")
                            questions = prompter.get_prompt(cluster_data, "q2", category)
                            
                            labels = [category] * len(questions)
                            rationales = []
                            for question in questions:
                                response_q2, _ = prompter.prompt("src/llm_eval/prompts/q2/simplified_system_prompt.txt", question, use_context=False)
                                label, score, rationale = extract_info_q1_q2(response_q2, get_label=False)
                                if score == None:
                                    import pdb; pdb.set_trace()
                                fit_data.append(score)
                                rationales.append(rationale)
                                                        
                        else: # prompt_mode == "q1_and_q2"
                            questions = prompter.get_prompt(cluster_data, "q1_q2")
                            
                            labels, rationales = [], []
                            for question in questions:
                                response_q2, _ = prompter.prompt("src/llm_eval/prompts/q1_q2/simplified_system_prompt.txt", question, use_context=False)
                                label, score, rationale = extract_info_q1_q2(response_q2, get_label=True)
                                labels.append(label)
                                fit_data.append(score)
                                rationales.append(rationale)
                
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
                    
    if llm_results_q2 == []:
        llm_results_q2 = None
    if llm_results_q3 == []:
        llm_results_q3 = None
        
    # Correlations with user study data and ground truth 
    agreement_by_topic = compute_agreement_per_topic(responses_by_id)
    corr_results = compute_correlations_one(corr_data, rank_llm_data=llm_results_q3, fit_llm_data=llm_results_q2)
                
    corr_results2 = compute_correlations_two(responses_by_id, llm_results_q3, llm_results_q2)

    # Print and save results
    logging.info("--Correlation results--")
    logging.info(corr_results)
    #logging.info(corr_results2)
    
    # store also the results of the prompts (llm_results_qX)
    # create a folder at path_save / {prompt_mode}_{llm_models}_{timestamp} with all the files
    logging.info("--Saving results--")
    path_save = "data/files_pilot"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_models = "_".join(model_types)
    path_save = f"{path_save}/{prompt_mode}_{llm_models}_{timestamp}"
    os.makedirs(path_save, exist_ok=True)
    corr_results.to_excel(f"{path_save}/correlation_results_mode1.xlsx", index=False)
    corr_results2.to_excel(f"{path_save}/correlation_results_mode2.xlsx", index=False)
    pd.DataFrame(llm_results_q1).to_excel(f"{path_save}/llm_results_q1.xlsx", index=False)
    if llm_results_q2:
        pd.DataFrame(llm_results_q2).to_excel(f"{path_save}/llm_results_q2.xlsx", index=False)
    if llm_results_q3:
        pd.DataFrame(llm_results_q3).to_excel(f"{path_save}/llm_results_q3.xlsx", index=False)
        
    # calculate npmi correlations
    columns_calculate = [col for col in corr_results.columns if col not in["id", "model", "topic", "n_annotators", "fit_rho", "fit_tau", "fit_agree"]]
    corr_results = calculate_corr_npmi(corr_results, npmi_data, columns_calculate)
    corr_results.to_excel(f"{path_save}/corrs_mode1_npmi.xlsx", index=False)
    
if __name__ == "__main__":
    main()
