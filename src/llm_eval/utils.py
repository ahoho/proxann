import json
import re
from collections import Counter
from copy import deepcopy
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr
from sklearn.metrics import ndcg_score

from irrCAC.raw import CAC


def extend_to_full_sentence(
    text: str,
    num_words: int
) -> str:
    """Truncate text to a certain number of words and extend to the end of the sentence so it's not cut off.
    """
    text_in_words = text.split()
    truncated_text = " ".join(text_in_words[:num_words])
    
    # Check if there's a period after the truncated text
    remaining_text = " ".join(text_in_words[num_words:])
    period_index = remaining_text.find(".")
    
    # If period, extend the truncated text to the end of the sentence
    if period_index != -1:
        extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
    else:
        extended_text = truncated_text
    
    # Clean up screwed up punctuations        
    extended_text = re.sub(r'\s([?.!,"])', r'\1', extended_text)
    
    return extended_text

def bradley_terry_model(
    pair_ids: List[dict[str, str]],
    orders: List[str],
    logprobs: Optional[List[float]] = None,
    num_iters: int = 1000,
    lr: float = 0.01
) -> pd.DataFrame:
    """
    Perform Bradley-Terry model for pairwise rank comparisons. If logprobs are provided, the learning rate is adjusted based on the logprobs.
    
    Parameters:
    ----------
    pair_ids: List[dict[str, str]]
        List of dictionaries containing the document IDs for each pair.
    orders: List[str]
        List of orders for each pair (A or B).
    logprobs: Optional[List[float]]
        List of log probabilities for each pair.
    num_iters: int
        Number of iterations for the model.
    lr: float
        Learning rate for the model.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with the ranked document IDs and scores, sorted by score in descending order.
    """
    # Extract document IDs and initialize their scores
    doc_ids = {doc_id for pair in pair_ids for doc_id in pair.values()}
    scores = {doc_id: 0.0 for doc_id in doc_ids} 

    pairwises = []
    for i, order in enumerate(orders):
        pair = pair_ids[i]
        if order == 'A':
            pairwises.append((pair["A"], pair["B"]))  # A wins over B
        elif order == 'B':
            pairwises.append((pair["B"], pair["A"]))  # B wins over A

    # Likelihood maximization
    for _ in range(num_iters):
        score_updates = {doc_id: 0.0 for doc_id in doc_ids}
        
        for i, (winner, loser) in enumerate(pairwises):
            # Calculate probabilities based on current scores
            score_diff = scores[winner] - scores[loser]
            probability_win = 1 / (1 + np.exp(-score_diff))  
            
            # Adjust lr if logprobs are provided (higher logprob means higher certainty)
            adjustment_factor = np.exp(logprobs[i]) if logprobs and i < len(logprobs) else 1.0
            adjusted_lr = lr * adjustment_factor
            
            # Winner gets stronger, loser gets weaker
            score_updates[winner] += adjusted_lr * (1 - probability_win)
            score_updates[loser] -= adjusted_lr * (1 - probability_win)
        
        for doc_id in doc_ids:
            scores[doc_id] += score_updates[doc_id]
    
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranked_df = pd.DataFrame(ranked_docs, columns=["doc_id", "score"])
    
    return ranked_df

def extract_info(text, get_label):
    """
    Extracts the label, order, and rationale from the prompt text based on the 'get_label' parameter. If 'get_label' is set to True, the method extracts from the 'q1_q3' prompt; otherwise, it extracts from 'q3'.
    """    
    label_pattern = r'Label:\s*(.*?)\s*(?=Order:|Comparison:)'
    order_pattern = r'Order:\s*(?:DOCUMENT\s*)?([AB])' 
    rationale_pattern = r'Comparison:\s*(.*)'

    if get_label: 
        label_match = re.findall(label_pattern, text, re.DOTALL)
        label = label_match[0].strip() if label_match else ""
    else:
        label = ""
        
    order_match = re.findall(order_pattern, text)
    order = order_match[0].strip() if order_match else ""
    
    rationale_match = re.findall(rationale_pattern, text, re.DOTALL)
    rationale = rationale_match[0].strip() if rationale_match else ""

    return label, order, rationale

def extract_logprobs(pairwise_logprobs, backend, logging):
    """Extracts log probabilities associated with the pairwise rankings (i.e., whether the more related document is A or B) from LLM responses, handling different backends (i.e., LLM types)."""
    
    if pairwise_logprobs is None:
        return None
    
    try:
        if backend == "llama_cpp":
            return [item['probs'][0]['prob'] for item in pairwise_logprobs if item['probs'][0]['tok_str'].strip() in {"A", "B"}]
        else:
            return [
                top_logprob.logprob
                for token_logprob in pairwise_logprobs
                for top_logprob in token_logprob.top_logprobs
                if top_logprob.token.strip() in {"A", "B"}
            ]
    except Exception as e:
        logging.error(f"-- -- Error extracting logprobs: {e}")
        return None

# ============================================================================ #
# Pilot study parsing functions
# ============================================================================ #
def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries at the second level."""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def load_config_pilot(config_paths):
    """Load pilot config files and merge them into a single dictionary at the second level."""
    config = {}
    for path in config_paths.split(","):
        with open(path, 'r') as f:
            new_config = json.load(f)
            config = merge_dicts(config, new_config)
            
    return config

def process_responses(
    response_csv: str,
    data_jsons: str, 
    min_minutes: int = 5, 
    start_date: str="2024-06-28 12:00:00",
    n_eval_docs: int = 7,
    path_save: str ="data/files_pilot"
):
    """
    Process responses from the pilot study. The function filters out disqualified responses and returns a dictionary of responses by id, where the id is in the form 'data/models/model_name/topic_id'.
    
    Parameters:
    ----------
    response_csv: str
        Path to the CSV file containing the Qualtrics responses (data/files_pilot/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv)
    data_jsons: List[str]
        Paths to the JSON files containing the evaluation data (data/files_pilot/config_first_round.json, data/files_pilot/config_second_round.json)
    min_minutes: int
        Minimum time cutoff for a response (default: 5)
    start_date: str
        Start date for the responses (default: "2024-06-28 12:00:00")
    n_eval_docs: int
        Number of evaluation documents (default: 7)
    """
    
    # Load response data
    raw_responses = pd.read_csv(response_csv)
    column_names = dict(zip(raw_responses.columns, raw_responses.iloc[0]))
    raw_responses = raw_responses.iloc[2:]  # skip header rows from Qualtrics
    raw_responses = raw_responses.loc[raw_responses["Status"] == "IP Address"]  # remove preview data
    raw_responses = raw_responses.loc[raw_responses["StartDate"] >= start_date]
    
    print(f"Total responses: {len(raw_responses)}")
    
    # Load evaluation data
    eval_data = {}
    topics_per_model = Counter()
    for json_fpath in data_jsons:
        with open(json_fpath) as infile:
            raw_eval_data = json.load(infile)
        
        for model_id, model_data in raw_eval_data.items():
            for cluster_id, cluster_data in model_data.items():
                cluster_data["topic_match_id"] = topics_per_model[model_id] # unique id for each topic within a model
                eval_data[f"{model_id}/{cluster_id}"] = cluster_data
                topics_per_model[model_id] += 1

    # Verify column names
    assert column_names["11_loop_fit_a"].startswith("Attention Check")
    assert column_names["1_loop_fit_a"].startswith("${e://Field/eval_doc_0}")
    assert column_names[f"{n_eval_docs}_loop_fit_a"].startswith(f"${{e://Field/eval_doc_{n_eval_docs-1}}}")
    assert column_names["rank_99"].endswith("distractor_doc")
    assert column_names["rank_0"].endswith("eval_doc_0")
    assert column_names[f"rank_{n_eval_docs-1}"].endswith(f"eval_doc_{n_eval_docs-1}")
    assert "prolific ID" in column_names["Q22"]

    # Parse responses
    responses = []
    time_cutoff = min(np.quantile(raw_responses["Duration (in seconds)"][2:].astype(float), 0.05), 60 * min_minutes)
    
    for _, row in raw_responses.iterrows():
        r = {}
        
        if str(row["rank_99"]) == "nan":
            continue
        
        r["cluster_id"] = row["id"]
        r["annotator_id"] = row["Q22"]
        cluster_data = eval_data[r["cluster_id"]]
        r["too_quick"] = float(row["Duration (in seconds)"]) < time_cutoff
        r["category"] = row["cluster_label"]
        r["failed_category"] = r["category"] in " ".join(cluster_data["topic_words"])
        r["failed_purpose"] = "single category" not in row["practice_purpose"]
        r["failed_fit_check"] = not row["11_loop_fit_a"].startswith("2")
        r["failed_fam_check"] = not str(row["11_loop_familiarity"]).startswith("I am not familiar")
        r["failed_sponge_check_strict"] = row["rank_99"] != "8"
        r["failed_sponge_check_weak"] = row["rank_99"] not in ["7", "8"]

        practice_ranks = [int(row[f"practice_rank_{i}"]) for i in range(4)]
        r["failed_practice_rank_strict"] = practice_ranks != [1, 2, 3, 4]
        r["failed_practice_rank_weak"] = practice_ranks[0] not in [1,2] or practice_ranks[3] != 4

        r["remove"] = (
            r["failed_purpose"] or r["failed_category"] or r["too_quick"] or
            r["failed_fit_check"] or r["failed_sponge_check_weak"] or r["failed_practice_rank_weak"] or
            (r["failed_fam_check"] and (r["failed_practice_rank_strict"] or r["failed_sponge_check_strict"]))
        )
        
        r["StartDate"] = row["StartDate"]
        r["time"] = float(row["Duration (in seconds)"])
        r["eval_docs"] = deepcopy(cluster_data["eval_docs"])
        r["exemplar_docs"] = cluster_data["exemplar_docs"]
        r["topic_words"] = cluster_data["topic_words"]
        r["topic_match_id"] = cluster_data["topic_match_id"]
        
        label = row["cluster_label"]
        clarity = int(row["cluster_coherence"].split("-")[0].strip())

        for i in range(n_eval_docs):
            fit_answer = int(row[f"{i+1}_loop_fit_a"].split("-")[0].strip())
            r["eval_docs"][i]["fit"] = fit_answer
            r["eval_docs"][i]["rank"] = int(row[f"rank_{i}"])
            r["eval_docs"][i]["is_familiar"] = not str(row[f"{i+1}_loop_familiarity"]).startswith("I am not familiar")
                    
        responses.append(r)
    
    # Filter out disqualified responses
    print(f"Total responses: {len(responses)}")
    print(f"Removed: {sum(r['remove'] for r in responses)}")
    
    responses = [r for r in responses if not r["remove"]]
    responses_by_id = {cluster_id: [] for cluster_id in eval_data.keys()}
    for r in responses:
        responses_by_id[r["cluster_id"]].append(r)
    
    # Save cluster rank counts
    counts = Counter([r["cluster_id"] for r in responses])
    with open(f"{path_save}/cluster_rank_counts.json", "w") as outfile:
        json.dump(counts, outfile, indent=2)

    return responses_by_id


def compute_agreement_per_topic(
    responses_by_id: dict[str, List[dict]],
    fit_threshold: int = 4,
    n_eval_docs: int = 7
):
    """
    Compute agreement metrics (Krippendorff's alpha and Gwet's AC2 for fit and rank data) for each topic.
    """
    agreement_data = []
    bin_fit_data_by_model = {"mallet": [], "ctm": [], "category-45": []}

    for topic_id, group in responses_by_id.items():
        if len(group) < 2:
            continue

        # Define indices for annotators and documents
        ann_idxs = [f"{topic_id}_ann_{i}" for i in range(len(group))]
        doc_idxs = [f"{topic_id}_doc_{i}" for i in range(len(group[0]["eval_docs"]))]
        
        # Collect fit and rank data for the group
        fit_data = pd.DataFrame(
            [[doc["fit"] for doc in r["eval_docs"]] for r in group],
            index=ann_idxs,
            columns=doc_idxs
        ).T
        bin_fit_data = (fit_data >= fit_threshold).astype(str).add_prefix(f"{topic_id}_")
        
        rank_data = pd.DataFrame(
            [[doc["rank"] for doc in r["eval_docs"]] for r in group],
            index=ann_idxs,
            columns=doc_idxs
        ).T
        
        # Calculate agreement metrics
        fit_cac = CAC(fit_data, weights="ordinal", categories=[1, 2, 3, 4, 5])
        bin_fit_cac = CAC(bin_fit_data, weights="identity", categories=[f"{topic_id}_True", f"{topic_id}_False"])
        rank_cac = CAC(rank_data, weights="ordinal", categories=list(range(1, n_eval_docs+1)))

        fit_alpha = fit_cac.krippendorff()["est"]
        fit_ac2 = fit_cac.gwet()["est"]

        rank_alpha = rank_cac.krippendorff()["est"]
        rank_ac2 = rank_cac.gwet()["est"]

        # Parse model name and topic from the topic ID
        split_id = topic_id.split("/")
        model_name = split_id[-2]
        topic = split_id[-1]

        # Append agreement data
        agreement_data.append({
            "id": topic_id,
            "model": model_name,
            "topic": topic,
            "fit_alpha": fit_alpha["coefficient_value"],
            "fit_alpha_p": fit_alpha["p_value"],
            "fit_ac2": fit_ac2["coefficient_value"],
            "fit_ac2_p": fit_ac2["p_value"],
            "rank_alpha": rank_alpha["coefficient_value"],
            "rank_alpha_p": rank_alpha["p_value"],
            "rank_ac2": rank_ac2["coefficient_value"],
            "rank_ac2_p": rank_ac2["p_value"],
        })

        # Collect binary fit data by model
        bin_fit_data_by_model[model_name].append(bin_fit_data)

    # Convert agreement data to a DataFrame
    agreement_data_by_topic = pd.DataFrame(agreement_data)

    return agreement_data_by_topic, bin_fit_data_by_model

def collect_fit_rank_data(
    responses_by_id: dict[str, list[dict]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    """Collect fit, rank, and probability data for each model."""
    
    # Initialize data dictionaries
    model_fit_data = {"mallet": [], "ctm": [], "category-45": []}
    model_rank_data = {"mallet": [], "ctm": [], "category-45": []}
    model_prob_data = {"mallet": [], "ctm": [], "category-45": []}
    corr_data = []

    # Process each topic's response data
    for topic_id, group in responses_by_id.items():
        model_name, topic = topic_id.split("/")[-2:]
        
        # Skip if there are fewer than 2 responses in the group
        if len(group) < 2:
            continue

        # Collect data matrices for fit, rank, probability, and assignments
        fit_data = np.array([[doc["fit"] for doc in r["eval_docs"]] for r in group])
        rank_data = 8 - np.array([[doc["rank"] for doc in r["eval_docs"]] for r in group])  # Reverse rank
        prob_data = np.array([doc["prob"] for doc in group[0]["eval_docs"]])
        assign_data = np.array([doc["assigned_to_k"] for doc in group[0]["eval_docs"]])

        # Append data for each model
        model_fit_data[model_name].append(fit_data)
        model_rank_data[model_name].append(rank_data)
        model_prob_data[model_name].append(np.repeat([prob_data], len(group), axis=0))
        
        # Record correlation data
        corr_data.append({
            "id": topic_id,
            "model": model_name,
            "n_annotators": len(group),
            "topic": topic,
            "topic_match_id": group[0]["topic_match_id"],
            "fit_data": fit_data,
            "rank_data": rank_data,
            "prob_data": prob_data,
            "assign_data": assign_data
        })
    
    return model_fit_data, model_rank_data, model_prob_data, corr_data

def compute_correlations(corr_data, corr_llm_data=None, aggregation_method="mean", fit_threshold=4):
    """Compute correlation coefficients for fit and rank data."""
    
    if corr_llm_data is None:
        corr_llm_data = [None] * len(corr_data)
        
    corr_results = []
    for data, llm_data in zip(corr_data, corr_llm_data):
        
        fit_data, rank_data, prob_data = data["fit_data"], data["rank_data"], data["prob_data"]
        
        annotator_results = {} 
        if llm_data is not None:
            rank_llm_data = llm_data["rank_data"]
            annotators = llm_data["annotators"]
        
        if aggregation_method == "mean":
            mean_fit_data, mean_rank_data = fit_data.mean(0), rank_data.mean(0)
        elif aggregation_method == "concatenate":
            mean_fit_data, mean_rank_data = fit_data.flatten(), rank_data.flatten()
            prob_data = np.tile(prob_data, len(fit_data))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Compute general correlations
        fit_rho, _ = spearmanr(mean_fit_data, prob_data)
        rank_rho, _ = spearmanr(mean_rank_data, prob_data)
        fit_tau, _ = kendalltau(mean_fit_data, prob_data)
        rank_tau, _ = kendalltau(mean_rank_data, prob_data)
        
        # Calculate fit agreement
        fit_agree = np.mean((fit_data >= fit_threshold).astype(int) == data["assign_data"])
        
        # LLM correlations if llm_data
        if llm_data is not None:
            for annotator, rank_llm in zip(annotators, rank_llm_data):
                rank_rho_users, _ = spearmanr(mean_rank_data, rank_llm)
                rank_rho_gt, _ = spearmanr(prob_data, rank_llm)
                rank_tau_users, _ = kendalltau(mean_rank_data, rank_llm)
                rank_tau_gt, _ = kendalltau(prob_data, rank_llm)
                
                # Add annotator-specific results to the dictionary
                annotator_results[f"rank_rho_users_{annotator}"] = rank_rho_users
                annotator_results[f"rank_rho_gt_{annotator}"] = rank_rho_gt
                annotator_results[f"rank_tau_users_{annotator}"] = rank_tau_users
                annotator_results[f"rank_tau_gt_{annotator}"] = rank_tau_gt

        corr_results.append({
            "id": data["id"],
            "model": data["model"],
            "topic": data["topic"],
            "n_annotators": data["n_annotators"],
            "fit_rho": fit_rho,
            "fit_tau": fit_tau,
            "rank_rho": rank_rho,
            "rank_tau": rank_tau,
            "fit_agree": fit_agree,
            **annotator_results 
        })
    
    return pd.DataFrame(corr_results)

def perform_mann_whitney_tests(corr_data):
    """Perform Mann-Whitney U tests between models for each metric."""
    #metrics = ["fit_rho", "rank_rho", "fit_tau", "rank_tau", "fit_agree"]
    no_include = ["id", "model", "topic", "n_annotators"]
    metrics = [col for col in corr_data.columns if col not in no_include]
    alpha = 0.05 / len(metrics)  # Bonferroni correction

    for model_a, model_b in combinations(corr_data["model"].unique(), 2):
        for metric in metrics:
            stat, pval = mannwhitneyu(
                corr_data[corr_data["model"] == model_a][metric].values,
                corr_data[corr_data["model"] == model_b][metric].values
            )
            sig = "*" if pval < alpha else ""
            print(f"{model_a} vs {model_b} | {metric} | {stat:.3f} | {pval:.3f}{sig}")

def compute_ndcg_scores(model_fit_data, model_rank_data, model_prob_data):
    """Compute NDCG scores for each model's fit and rank data."""
    ndcg_results = []

    for model_name in model_fit_data.keys():
        fit_data = np.concatenate(model_fit_data[model_name], axis=0)
        rank_data = np.concatenate(model_rank_data[model_name], axis=0)
        prob_data = np.concatenate(model_prob_data[model_name], axis=0)

        fit_ndcg = ndcg_score(fit_data, prob_data)
        rank_ndcg = ndcg_score(rank_data, prob_data)

        print(f"{model_name} | Fit NDCG: {fit_ndcg:.5f} | Rank NDCG: {rank_ndcg:.5f}")
        ndcg_results.append({
            "model": model_name,
            "fit_ndcg": fit_ndcg,
            "rank_ndcg": rank_ndcg,
        })

    return pd.DataFrame(ndcg_results)