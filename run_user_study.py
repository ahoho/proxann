import argparse
from collections import Counter
import logging
import datetime
from src.llm_eval.prompter import Prompter
from src.llm_eval.utils import bradley_terry_model, collect_fit_rank_data, compute_correlations_one, compute_correlations_two, extract_info_q1_q3, extract_logprobs, load_config_pilot, process_responses, extract_info_q1_q2

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
    config_pilot = load_config_pilot(args.config_path)

    model_types = args.model_type.split(",") if args.model_type else []
    prompt_modes = args.prompt_mode.split(",") if args.prompt_mode else []
    llm_results_q3, llm_results_q2 = [], []
    topics_per_model = Counter()
    
      
    # keep dictionary with only first key (for debugging)
    """
    inside_dict = {
        '32':  config_pilot['data/models/mallet']['32']
    }
    config_pilot = {
        'data/models/mallet': inside_dict
    }
    """
    for prompt_mode in prompt_modes:
        logging.info(f"-- -- Executing in MODE: {prompt_mode} -- --")
        for model_id, model_data in config_pilot.items():
            logging.info(f"-- -- Model: {model_id}")
            for cluster_id, cluster_data in model_data.items():
                id_ = f"{model_id}/{cluster_id}"
                model = model_id.split("/")[-1]
                topic_match_id = topics_per_model[model_id]
                rank_data = []

                for llm_model in model_types:
                    logging.info(f"-- -- -- -- LLM: {llm_model}")
                    prompter = Prompter(model_type=llm_model)
                    
                    if prompt_mode == "q1_then_q3" or prompt_mode == "q1_and_q3":

                        if prompt_mode == "q1_then_q3":
                            logging.info("-- Executing Q1...")
                            question = prompter.get_prompt(cluster_data, "q1")
                            category, _ = prompter.prompt("src/llm_eval/prompts/q1/simplified_system_prompt.txt", question, use_context=False)
                            logging.info("-- Executing Q3...")
                            questions, pair_ids = prompter.get_prompt(cluster_data, "q3", category)
                            get_label = False

                        elif prompt_mode == "q1_and_q3":
                            questions, pair_ids = prompter.get_prompt(cluster_data, "q1_q3")
                            get_label = True

                        labels, orders, rationales, logprobs = [], [], [], []
                        for question in questions:
                            pairwise, pairwise_logprobs = prompter.prompt(
                                "src/llm_eval/prompts/q1_q3/simplified_system_prompt.txt" if prompt_mode == "q1_and_q3" else "src/llm_eval/prompts/q3/simplified_system_prompt.txt",
                                question, use_context=False)

                            try:
                                label, order, rationale = extract_info_q1_q3(pairwise, get_label=get_label)
                                labels.append(label)
                                orders.append(order)
                                rationales.append(rationale)
                            except Exception as e:
                                logging.error(f"-- -- Error extracting info from prompt: {e}")

                            # Extract logprobs if available
                            if pairwise_logprobs is not None:
                                prob_values = extract_logprobs(pairwise_logprobs, prompter.backend,logging)
                                if prob_values:
                                    logprobs.append(prob_values[0])
                                else:
                                    logging.warning(f"-- -- No logprobs extracted for model {model_id} and cluster {cluster_id}")

                        if not logprobs:
                            logprobs = None
                            logging.warning(f"-- -- No logprobs found for model {model_id} and cluster {cluster_id}")

                        # Obtain full rank (Bradley-Terry model)
                        ranked_documents = bradley_terry_model(pair_ids, orders, logprobs) #, use_choix=False
                        true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
                        rank = [true_order.index(doc_id) + 1 for doc_id in ranked_documents['doc_id']]
                        rank = [len(rank) - r + 1 for r in rank] # Invert rank
                        rank_data.append(rank)

                        llm_results_q3.append({
                            "id": id_,
                            "model": model,
                            "n_annotators": len(model_types),
                            "annotators": model_types,
                            "topic": cluster_id,
                            "topic_match_id": topic_match_id,
                            "rank_data": rank_data
                        })
                        
                    elif prompt_mode == "q1_then_q2" or prompt_mode == "q1_and_q2":
                        if prompt_mode == "q1_and_q2":
                            
                            questions = prompter.get_prompt(cluster_data, "q1_q2")
                            
                            labels, scores, rationales = [], [], []
                            for question in questions:
                                response_q2, _ = prompter.prompt("src/llm_eval/prompts/q1_q2/simplified_system_prompt.txt", question, use_context=False)
                                label, score, rationale = extract_info_q1_q2(response_q2, get_label=True)
                                labels.append(label)
                                scores.append(score)
                                rationales.append(rationale)
                                
                        else: #q1_then_q2
                            logging.info("-- Executing Q1...")
                            question = prompter.get_prompt(cluster_data, "q1")
                            category, _ = prompter.prompt("src/llm_eval/prompts/q1/simplified_system_prompt.txt", question, use_context=False)
                            
                            logging.info("-- Executing Q2...")
                            questions = prompter.get_prompt(cluster_data, "q2", category)
                            
                            labels = [category] * len(questions)
                            _, scores, rationales = [], [], []
                            for question in questions:
                                response_q2, _ = prompter.prompt("src/llm_eval/prompts/q2/simplified_system_prompt.txt", question, use_context=False)
                                label, score, rationale = extract_info_q1_q2(response_q2, get_label=False)
                                scores.append(score)
                                rationales.append(rationale)
                        
                        llm_results_q2.append({
                            "id": id_,
                            "model": model,
                            "n_annotators": len(model_types),
                            "annotators": model_types,
                            "topic": cluster_id,
                            "topic_match_id": topic_match_id,
                            "labels": labels,
                            "fit_data": [scores],
                        })
                        
    if llm_results_q2 == []:
        llm_results_q2 = None
    if llm_results_q3 == []:
        llm_results_q3 = None
        
    # Correlations with user study data and ground truth    
    responses_by_id = process_responses(args.response_csv, args.config_path.split(","))
    _, _, _, corr_data = collect_fit_rank_data(responses_by_id)
    corr_results = compute_correlations_one(corr_data, rank_llm_data=llm_results_q3, fit_llm_data=llm_results_q2)
    corr_results2 = compute_correlations_two(responses_by_id, llm_results_q3, llm_results_q2)
    
    # Print and save results
    logging.info("--Correlation results--")
    logging.info(corr_results)
    logging.info(corr_results2)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_models = "_".join(model_types)
    corr_results.to_excel(f"data/files_pilot/correlation_results_mode1_{prompt_mode}_{llm_models}_{timestamp}.xlsx", index=False)
    corr_results2.to_excel(f"data/files_pilot/correlation_results_mode2_{prompt_mode}_{llm_models}_{timestamp}.xlsx", index=False)

    
if __name__ == "__main__":
    main()
