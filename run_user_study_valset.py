import argparse
import logging
import itertools
import numpy as np
import pandas as pd
from src.llm_eval.prompter import Prompter
from src.llm_eval.utils import collect_fit_rank_data, extract_info_binary_q2, extract_info_q1_q3, load_config_pilot, process_responses, extract_logprobs

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
    
    model_types = args.model_type.split(",") if args.model_type else []
    prompt_modes = args.prompt_mode.split(",") if args.prompt_mode else []
    
    # create validation data
    logging.info("-- -- Creating validation data")
    """
    doc_pairs = []
    for topic_id, topic_responses in responses_by_id.items():
        eval_docs = topic_responses[0]["eval_docs"]
        ranks = 8 - np.array([[d["rank"] for d in resp["eval_docs"]] for resp in topic_responses])
        fits = np.array([[d["fit"] for d in resp["eval_docs"]] for resp in topic_responses])
        labels = [d["category"] for d in topic_responses]
        for i, j in itertools.combinations(range(ranks.shape[1]), 2):
            rank_diffs = ranks[:, i] - ranks[:, j]
            fit_diffs = fits[:, i] - fits[:, j]
            
            # compute agreement (allow for ties)
            ranks_agree = np.all(rank_diffs > 0) | np.all(rank_diffs < 0)
            fits_agree = np.all(fit_diffs >= 0) | np.all(fit_diffs <= 0)
            ranks_mean = np.mean(rank_diffs)
            fits_mean = np.mean(fit_diffs)
            rank_and_fit_agree = ranks_agree and fits_agree and (np.sign(ranks_mean) == np.sign(fits_mean))

            if rank_and_fit_agree:
                # get the winner of the pair
                winner = eval_docs[j]["doc_id"] if fits_mean < 0 else eval_docs[i]["doc_id"]
            else:
                winner = -1
            doc_pairs.append({
                "topic_id": topic_id,
                "doc_id1": eval_docs[i]["doc_id"],
                "doc_id2": eval_docs[j]["doc_id"],
                "doc1": eval_docs[i]["text"],
                "doc2": eval_docs[j]["text"],
                "rank_diffs_mean": ranks_mean,
                "fit_diffs_mean": fits_mean,
                "ranks_agree": ranks_agree,
                "fits_agree": fits_agree,
                "rank_and_fit_agree": rank_and_fit_agree,
                "user_labels": labels,
                "users_winner": winner
            })

    doc_pairs = pd.DataFrame(doc_pairs)
    doc_pairs["mean_of_means"] = (np.abs(doc_pairs["rank_diffs_mean"]) + np.abs(doc_pairs["fit_diffs_mean"])) / 2
    doc_pairs = doc_pairs.sort_values("mean_of_means", ascending=False)
    unambiguous_doc_pairs = doc_pairs[doc_pairs["rank_and_fit_agree"]]
    """
    
    doc_pairs = []
    for topic_id, topic_responses in responses_by_id.items():
        eval_docs = topic_responses[0]["eval_docs"]
        ranks = 8 - np.array([[d["rank"] for d in resp["eval_docs"]] for resp in topic_responses])
        fits = np.array([[d["fit"] for d in resp["eval_docs"]] for resp in topic_responses])
        labels = [d["category"] for d in topic_responses]
        for i, j in itertools.combinations(range(ranks.shape[1]), 2):
            rank_diffs = ranks[:, i] - ranks[:, j]
            fit_diffs = fits[:, i] - fits[:, j]
            
            # compute agreement (allow for ties)
            ranks_agree = np.all(rank_diffs > 0) | np.all(rank_diffs < 0)
            fits_agree = np.all(fit_diffs >= 0) | np.all(fit_diffs <= 0)
            ranks_mean = np.mean(rank_diffs)
            fits_mean = np.mean(fit_diffs)
            rank_and_fit_agree = ranks_agree and fits_agree and (np.sign(ranks_mean) == np.sign(fits_mean))
            
            rank_mv = np.sign(np.mean(np.sign(rank_diffs)))
            fit_mv = np.sign(np.mean(np.sign(fit_diffs)))
            
            if rank_and_fit_agree:
                # get the winner of the pair
                rank_winner = eval_docs[j]["doc_id"] if fits_mean < 0 else eval_docs[i]["doc_id"]
                fit_winner = eval_docs[j]["doc_id"] if fits_mean < 0 else eval_docs[i]["doc_id"]
            else:
                # Choose higher-rated pair based on rank_mv
                if rank_mv > 0:
                    rank_winner = eval_docs[i]["doc_id"]  # i has higher rating
                elif rank_mv < 0:
                    rank_winner = eval_docs[j]["doc_id"]  # j has higher rating
                else:
                    rank_winner = -1
                    print("there is a tie")
                
                # Choose higher-rated pair based on fit_mv
                if fit_mv > 0:
                    fit_winner = eval_docs[i]["doc_id"]  # i has higher rating
                elif fit_mv < 0:
                    fit_winner = eval_docs[j]["doc_id"]  # j has higher rating
                else:
                    fit_winner = -1
                    print("there is a tie")
                
            doc_pairs.append({
                "topic_id": topic_id,
                "doc_id1": eval_docs[i]["doc_id"],
                "doc_id2": eval_docs[j]["doc_id"],
                "doc1": eval_docs[i]["text"],
                "doc2": eval_docs[j]["text"],
                "doc_id1_bin_fit": int(np.mean(fits[:, i]) >= 4),
                "doc_id2_bin_fit": int(np.mean(fits[:, j]) >= 4),
                "rank_diffs_mean": ranks_mean,
                "fit_diffs_mean": fits_mean,
                "ranks_agree": ranks_agree,
                "fits_agree": fits_agree,
                "rank_and_fit_agree": rank_and_fit_agree,
                "user_labels": labels,
                "users_rank_winner": rank_winner,
                "users_fit_winner": fit_winner
            })

    doc_pairs = pd.DataFrame(doc_pairs)
    doc_pairs["mean_of_means"] = (np.abs(doc_pairs["rank_diffs_mean"]) + np.abs(doc_pairs["fit_diffs_mean"])) / 2
    doc_pairs = doc_pairs.sort_values("mean_of_means", ascending=False)
    
    logging.info("-- -- Validation data created.")
    
    """
    do_ambiguous = False
    if do_ambiguous:
        # ambiguous_doc_pairs are when rank_and_fit_agree = False
        ambiguous_doc_pairs = doc_pairs[~doc_pairs["rank_and_fit_agree"]]
        
        # remove ties for the moment (we could allow the model to answer either way and mark it right)
        #ambiguous_doc_pairs = ambiguous_doc_pairs[ambiguous_doc_pairs["users_winner"] != -1]
    else:
        unambiguous_doc_pairs = doc_pairs[doc_pairs["rank_and_fit_agree"]]
        ambiguous_doc_pairs = unambiguous_doc_pairs
    """
    ambiguous_doc_pairs = doc_pairs
    
    # getting info from model
    for prompt_mode in prompt_modes:
        logging.info(f"-- -- Executing in MODE: {prompt_mode} -- --")
        for model_id, model_data in config_pilot.items(): 
            logging.info(f"-- -- Model: {model_id}")
            for cluster_id, cluster_data in model_data.items(): 
                logging.info(f"-- -- Cluster: {cluster_id}")
                id_ = f"{model_id}/{cluster_id}"
                
                if id_ in ambiguous_doc_pairs.topic_id.values:
                    
                    # keep unambiguous pairs from this model
                    ambiguous_doc_pairs_ = ambiguous_doc_pairs[ambiguous_doc_pairs.topic_id == id_]
                    
                    for llm_model in model_types:
                        logging.info(f"-- -- -- -- LLM: {llm_model}")
                        prompter = Prompter(model_type=llm_model)
                    
                        #==============================================
                        # Q1
                        #==============================================
                        logging.info("-- Executing Q1...")        
                        question = prompter.get_prompt(cluster_data, "q1")
                        category, _ = prompter.prompt("src/llm_eval/prompts/q1/simplified_system_prompt.txt", question, use_context=False)
                        print(f"\033[94mModel category: {category}\033[0m")
                        #==============================================
                        # Q2 (binary)
                        #==============================================
                        do_q2_with_q1_fixed = args.prompt_mode == "q1_then_q2_fix_cat"
                        q2_out = prompter.get_prompt(cluster_data, "binary_q2", category, do_q2_with_q1_fixed=do_q2_with_q1_fixed)
                        #==============================================
                        # Q3
                        #==============================================
                        q3_out = prompter.get_prompt(cluster_data, "q3", category, doing_both_ways=False, do_q3_with_q1_fixed=True)
                        get_label = False
                        
                        questions, pair_ids = q3_out
                        
                        this_cluster_all_idx = {cluster_data["doc_id"]: idx for idx, cluster_data in enumerate(cluster_data['eval_docs'])}
                            
                        for una_idx, una_pair in ambiguous_doc_pairs_.iterrows():   
                            if args.prompt_mode == "q1_then_binary_q2" or args.prompt_mode == "q1_then_q2_fix_cat":
                                logging.info("-- Executing Q2...")
                                for this_doc_idx, doc_id in enumerate([una_pair.doc_id1, una_pair.doc_id2]):
                                    # execute q2 on the pairs
                                    fit, logprobs = prompter.prompt("src/llm_eval/prompts/binary_q2/simplified_system_prompt.txt", q2_out[this_cluster_all_idx[doc_id]], use_context=False)

                                    ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_doc_id{this_doc_idx+1}_fit"] = extract_info_binary_q2(fit)
                                    
                                    if not fit.lower().endswith("yes") and not fit.lower().endswith("no"):
                                        import pdb; pdb.set_trace()
                                    else:
                                        logprob, top_logprobs = extract_logprobs(logprobs, llm_model, logging)
                                    
                                    ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_doc_id{this_doc_idx+1}_fit_logprobs"] = logprob
                                    ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_doc_id{this_doc_idx+1}_fit_top_logprobs"] = str(top_logprobs)

                            if args.prompt_mode == "q1_then_q3_fix_cat":        
                                logging.info("-- Executing Q3...")
                                # execute Q3 on the pairs
                                target_pairs = [{'A': una_pair.doc_id1, 'B': una_pair.doc_id2}, {'A': una_pair.doc_id2, 'B': una_pair.doc_id1}]
                                id_q = [index for index, d in enumerate(pair_ids) if d in target_pairs][0]
                                
                                question = questions[id_q]
                                
                                pairwise, pairwise_logprobs = prompter.prompt("src/llm_eval/prompts/q1_then_q3_fix_cat/simplified_system_prompt.txt", question, use_context=False)
                                _, order, _ = extract_info_q1_q3(pairwise, get_label=get_label)
                                if order != "":
                                    closest_llm = int(pair_ids[id_q]["A"] if order == "A" else pair_ids[id_q]["B"])
                                else:
                                    import pdb; pdb.set_trace()
                                    closest_llm = -1
                                
                                # extract logprobs
                                if not pairwise.endswith("A") and not pairwise.endswith("B"):
                                    import pdb; pdb.set_trace()
                                else:
                                    logprob, top_logprobs = extract_logprobs(pairwise_logprobs, llm_model, logging)
                                    
                                # save result in df
                                ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_rank_winner"] = closest_llm
                                # save also the label
                                ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_label"] = category
                                # save logprobs
                                ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_rank_logprobs"] = logprob
                                ambiguous_doc_pairs.loc[una_idx, f"{llm_model}_rank_top_logprobs"] = str(top_logprobs)
                            
    # calculate accuracy for each llm_model (llm_model_winner == users_winner)
    for llm_model in model_types:
        if f"{llm_model}_rank_winner" in ambiguous_doc_pairs.columns:
            ambiguous_doc_pairs[f"{llm_model}_correct"] = ambiguous_doc_pairs[f"{llm_model}_rank_winner"] == ambiguous_doc_pairs["users_rank_winner"]
            logging.info(f"Rank accuracy for {llm_model}: {ambiguous_doc_pairs[f'{llm_model}_correct'].mean()}")
        if f"{llm_model}_doc_id1_fit" in ambiguous_doc_pairs.columns:
            ambiguous_doc_pairs[f"{llm_model}_doc_id1_fit_correct"] = ambiguous_doc_pairs[f"{llm_model}_doc_id1_fit"] == ambiguous_doc_pairs["doc_id1_bin_fit"]
            ambiguous_doc_pairs[f"{llm_model}_doc_id2_fit_correct"] = ambiguous_doc_pairs[f"{llm_model}_doc_id2_fit"] == ambiguous_doc_pairs["doc_id2_bin_fit"]
            logging.info(f"Fit accuracy for {llm_model} docid1: {ambiguous_doc_pairs[f'{llm_model}_doc_id1_fit_correct'].mean()}")
            logging.info(f"Fit accuracy for {llm_model} docid2: {ambiguous_doc_pairs[f'{llm_model}_doc_id2_fit_correct'].mean()}")
            
    # save to json instead
    name_save = f"ambiguous_unambiguous_{args.prompt_mode}.json" #if do_ambiguous else f"unambiguous_{args.prompt_mode}.json"
    ambiguous_doc_pairs.to_json(name_save)       
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    main()