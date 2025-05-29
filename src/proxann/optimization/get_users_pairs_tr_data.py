import argparse
import itertools
import logging
import numpy as np
import pandas as pd
from utils import process_responses

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="data/files_pilot/config_first_round.json,data/files_pilot/config_second_round.json")
    parser.add_argument("--response_csv", type=str, default="data/files_pilot/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv")
    parser.add_argument("--mode", type=str, default="q3")
    return parser.parse_args()

def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Read results from user study
    responses_by_id = process_responses(args.response_csv, args.config_path.split(","))
    
    if args.mode == "q3":

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
                    
                if rank_and_fit_agree:
                    # get the winner of the pair
                    rank_winner = eval_docs[j]["doc_id"] if fits_mean < 0 else eval_docs[i]["doc_id"]
                else:
                    # Choose higher-rated pair based on rank_mv
                    if rank_mv > 0:
                        rank_winner = eval_docs[i]["doc_id"]  # i has higher rating
                    elif rank_mv < 0:
                        rank_winner = eval_docs[j]["doc_id"]  # j has higher rating
                    else:
                        rank_winner = -1
                        print("there is a tie")
                
                # generate one training sample for each user category
                for user_cat in labels:
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
                        "users_rank_winner": rank_winner,
                        "category": user_cat,
                        "exemplar_docs": topic_responses[0]["exemplar_docs"],
                        "topic_words": topic_responses[0]["topic_words"],
                    })

        doc_pairs = pd.DataFrame(doc_pairs)
        doc_pairs["mean_of_means"] = (np.abs(doc_pairs["rank_diffs_mean"]) + np.abs(doc_pairs["fit_diffs_mean"])) / 2
        # Save the pairs to a json
        doc_pairs.to_json("data/files_pilot/user_pairs_rank_tr_data.json")
    
    elif args.mode == "q2":
        doc_pairs = []
        for topic_id, topic_responses in responses_by_id.items():
            
            eval_docs = topic_responses[0]["eval_docs"]
            fits = np.array([[d["fit"] for d in resp["eval_docs"]] for resp in topic_responses])
            
            mv_binarized_fit_data = (np.round(np.mean(fits, axis=0)) >= 4).astype(int)
            labels = [d["category"] for d in topic_responses]
            
            # for each user category there will be as many training samples as there are eval_docs
            for user_cat in labels: 
                for doc, mv_fit in zip(eval_docs, mv_binarized_fit_data):
                    doc_pairs.append({
                        "topic_id": topic_id,
                        "doc_id": doc["doc_id"],
                        "doc": doc["text"],
                        "bin_fit": mv_fit,
                        "user_label": user_cat,
                    })
        
        doc_pairs = pd.DataFrame(doc_pairs)
        # Save the pairs to a json
        doc_pairs.to_json("data/files_pilot/user_fit_tr_data.json")
        import pdb; pdb.set_trace()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
if __name__ == "__main__":
    main()