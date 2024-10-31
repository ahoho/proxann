import argparse
from collections import Counter
import logging
from src.llm_eval.prompter import Prompter
from src.llm_eval.utils import bradley_terry_model, collect_fit_rank_data, compute_correlations, extract_info, extract_logprobs, load_config_pilot, perform_mann_whitney_tests, process_responses

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama3.2")
    parser.add_argument("--prompt_mode", type=str, choices=["q1_then_q3", "q1_and_q3"], default="q1_and_q3")
    parser.add_argument("--config_path", type=str, default="data/files_pilot/config_first_round.json,data/files_pilot/config_second_round.json")
    parser.add_argument("--response_csv", type=str, default="data/files_pilot/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv")
    return parser.parse_args()

def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    config_pilot = load_config_pilot(args.config_path)

    model_types = args.model_type.split(",") if args.model_type else []
    llm_results = []
    topics_per_model = Counter()

    logging.info(f"-- -- Executing in MODE: {args.prompt_mode} -- --")
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

                if args.prompt_mode == "q1_then_q3":
                    logging.info("-- Executing Q1...")
                    question = prompter.get_prompt(cluster_data, "q1")
                    category, _ = prompter.prompt("src/llm_eval/prompts/q1/system_prompt.txt", question, use_context=False)
                    
                    logging.info("-- Executing Q3...")
                    questions, pair_ids = prompter.get_prompt(cluster_data, "q3", category)
                    get_label = False

                elif args.prompt_mode == "q1_and_q3":
                    questions, pair_ids = prompter.get_prompt(cluster_data, "q1_q3")
                    get_label = True

                labels, orders, rationales, logprobs = [], [], [], []
                for question in questions:
                    pairwise, pairwise_logprobs = prompter.prompt(
                        "src/llm_eval/prompts/q1_q3/system_prompt.txt" if args.prompt_mode == "q1_and_q3" else "src/llm_eval/prompts/q3/system_prompt.txt",
                        question, use_context=False)

                    try:
                        label, order, rationale = extract_info(pairwise, get_label=get_label)
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
                ranked_documents = bradley_terry_model(pair_ids, orders, logprobs)
                true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
                rank = [true_order.index(doc_id) + 1 for doc_id in ranked_documents['doc_id']]
                rank = [len(rank) - r + 1 for r in rank] # Invert rank
                rank_data.append(rank)

            llm_results.append({
                "id": id_,
                "model": model,
                "n_annotators": len(model_types),
                "annotators": model_types,
                "topic": cluster_id,
                "topic_match_id": topic_match_id,
                "rank_data": rank_data
            })
    
    # Correlations with user study data and ground truth    
    responses_by_id = process_responses(args.response_csv, args.config_path.split(","))
    _, _, _, corr_data = collect_fit_rank_data(responses_by_id)
    corr_results = compute_correlations(corr_data, llm_results)
    
    # Print and save results
    logging.info("--Correlation results--")
    logging.info(corr_results)

    corr_results.to_excel(f"data/files_pilot/correlation_results_{args.prompt_mode}.xlsx", index=False)
    
if __name__ == "__main__":
    main()
