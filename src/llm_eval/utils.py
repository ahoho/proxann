import itertools
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import List
import choix
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau, spearmanr, pearsonr
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
    pair_ids: list[dict[str, str]],
    orders: list[str],
    logprobs: list[float] = None,
    num_iters: int = 1000,
    lr: float = 0.01,
    use_choix: bool = True
) -> pd.DataFrame:
    """
    Perform Bradley-Terry model for pairwise rank comparisons. If use_choix is True, the choix library is used to compute the rankings **without considering logprobs**.
    
    Parameters:
    ----------
    pair_ids: list[dict[str, str]]
        List of dictionaries containing the document IDs for each pair.
    orders: list[str]
        List of orders for each pair ('A' or 'B').
    logprobs: list[float], optional
        List of log probabilities for each pair.
    num_iters: int, optional
        Number of iterations for the model.
    lr: float, optional
        Learning rate for the model.
    use_choix: bool, optional
        Flag to determine whether to use the choix library for ranking.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with the ranked document IDs and scores, sorted by score in descending order.
    """
    
    # Extract unique document IDs
    doc_ids = {doc_id for pair in pair_ids for doc_id in pair.values()}
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    index_to_doc_id = {idx: doc_id for doc_id, idx in doc_id_to_index.items()}
    n_items = len(doc_ids)
    
    if use_choix:
        # Prepare data for choix
        data = []
        for i, order in enumerate(orders):
            pair = pair_ids[i]
            if order == 'A':
                data.append((doc_id_to_index[pair["A"]], doc_id_to_index[pair["B"]]))  # A wins over B
            elif order == 'B':
                data.append((doc_id_to_index[pair["B"]], doc_id_to_index[pair["A"]]))  # B wins over A
        
        # Compute parameters using choix's MM algorithm
        try:
            params = choix.ilsr_pairwise(n_items, data, max_iter=1000)
            
        except Exception as e:
            print(e)
            print("-- -- Graph is not strongly connected. Printing adjacency matrix:")
            graph = nx.DiGraph(data=data)
            graph.add_edges_from(data)
            adjacency_matrix = nx.to_numpy_array(graph, dtype=int)
            print(adjacency_matrix)
            print("-- -- Using a small regularization factor to solve the issue.--")
            params = choix.ilsr_pairwise(n_items, data, alpha=0.001) # adding a little bit of regularization when the comparison graph is not connected,
            
        # Convert parameters to a DataFrame
        ranked_docs = sorted(
            ((index_to_doc_id[idx], score) for idx, score in enumerate(params)),
            key=lambda item: item[1],
            reverse=True
        )
        ranked_df = pd.DataFrame(ranked_docs, columns=["doc_id", "score"])
    else:
        # Initialize scores
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
                
                # Adjust learning rate if logprobs are provided
                adjustment_factor = np.exp(logprobs[i]) if logprobs and i < len(logprobs) else 1.0
                adjusted_lr = lr * adjustment_factor
                
                # Update scores
                score_updates[winner] += adjusted_lr * (1 - probability_win)
                score_updates[loser] -= adjusted_lr * (1 - probability_win)
            
            for doc_id in doc_ids:
                scores[doc_id] += score_updates[doc_id]
        
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_df = pd.DataFrame(ranked_docs, columns=["doc_id", "score"])
    
    return ranked_df

def extract_info_q1_q3(text, get_label):
    """
    Extracts the label, order, and rationale from the prompt text based on the 'get_label' parameter. If 'get_label' is set to True, the method extracts from the 'q1_q3' prompt; otherwise, it extracts from 'q3'.
    """   
    label_pattern = r'LABEL:\s*(.*?)\s*(?=CLOSEST:|RATIONALE:)'
    #order_pattern = r'(?<!\w)CLOSEST(?!\w)\W*[:\-]?\W*(?:DOCUMENT\s*)?([AB])'
    order_pattern = r'(?<!\w)(?:I would select|CLOSEST|The closest document to)\W*[:\-]?\W*(?:DOCUMENT\s*)?([AB])'

    rationale_pattern = r'\bRATIONALE\b\W*:\W*(.*)'
    
    if get_label: 
        label_match = re.findall(label_pattern, text, re.DOTALL | re.IGNORECASE)
        label = label_match[0].strip() if label_match else ""
    else:
        label = ""
        
    order_match = re.findall(order_pattern, text, re.IGNORECASE)
    order = order_match[0].strip() if order_match else ""
    
    rationale_match = re.findall(rationale_pattern, text, re.DOTALL | re.IGNORECASE)
    rationale = rationale_match[0].strip() if rationale_match else ""

    return label, order, rationale
    
def extract_info_q1_q2(text, get_label):
    """
    Extracts the label, score, and rationale from the prompt text based on the 'get_label' parameter. 
    If 'get_label' is set to True, the method extracts from the 'q1_q2' prompt; otherwise, it extracts from 'q2'.
    """
    label_pattern = r'LABEL:\s*(.*?)\s*(?=SCORE:)'
    score_pattern = r'SCORE:\s*(\d+)'
    rationale_pattern = r'RATIONALE:\s*(.*)'

    if get_label: 
        label_match = re.findall(label_pattern, text, re.DOTALL)
        label = label_match[0].strip() if label_match else ""
    else:
        label = ""
    
    score_match = re.findall(score_pattern, text)
    score = score_match[0].strip() if score_match else ""
    try:
        score = int(score)
    except ValueError:
        score = None
    
    rationale_match = re.findall(rationale_pattern, text, re.DOTALL)
    rationale = rationale_match[0].strip() if rationale_match else ""
    
    return label, score, rationale

def extract_info_binary_q2(text):
    fit_pattern_general = r'FIT:\s*(YES|NO)' 
    fit_pattern_bracketed = r'\[\[ ## FIT ## \]\]\s*(YES|NO)'  # for dspy prompts

    fit_match = re.findall(fit_pattern_bracketed, text, re.DOTALL)
    if not fit_match:
        fit_match = re.findall(fit_pattern_general, text, re.DOTALL)

    fit = fit_match[0].strip() if fit_match else ""
    fit = int("yes" in fit.lower())
    
    return fit

def extract_logprobs(pairwise_logprobs, backend, logging):
    """Extracts log probabilities associated with the pairwise rankings (i.e., whether the more related document is A or B) from LLM responses, handling different backends (i.e., LLM types)."""
    
    if pairwise_logprobs is None:
        return None
    
    try:
        if backend == "llama_cpp":
            return [item['probs'][0]['prob'] for item in pairwise_logprobs if item['probs'][0]['tok_str'].strip() in {"A", "B"}]
        else:
            #return [
            #    top_logprob.logprob
            #    for token_logprob in pairwise_logprobs
            #    for top_logprob in token_logprob.top_logprobs
            #    if top_logprob.token.strip() in {"A", "B"}
            #]
            logprob = pairwise_logprobs[-1].logprob
            top_logprobs = pairwise_logprobs[-1].top_logprobs
            top_logprobs = [(top_logprob.token, top_logprob.logprob) for top_logprob in top_logprobs]
            
            return logprob, top_logprobs
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

def normalize_key(key, valid_models, valid_datasets, include_id=True):
    """
    Convert a cluster_id key to a normalized key using the following format: 
    '{dataset_name}/{model_name}/{topic_id}'.
    
    If include_id is set to False, the topic_id is omitted:
    '{dataset_name}/{model_name}'.
    """
    
    # We use a different pattern depending on whether we want to include the topic ID or not (from run_user_study.py we need it without)
    pattern = rf"/({'|'.join(map(re.escape, valid_models))})"
    if include_id:
        pattern += r"/.*?(\d+)$"
    else:
        pattern += r"(/|$)"
    
    # Model (and topic ID)
    match = re.search(pattern, key)
    if not match:
        raise ValueError(f"Key '{key}' does not contain a valid model" + (" or ID" if include_id else ""))

    model = match.group(1)
    id_part = match.group(2) if include_id else None

    # Dataset key
    dataset = next((ds for ds in valid_datasets if ds in key), None)
    if not dataset:
        raise ValueError(f"Key '{key}' does not contain a valid dataset")

    return f"{dataset}/{model}" + (f"/{id_part}" if include_id else "")

def process_responses(
    response_csv: str,
    data_jsons: str, 
    min_minutes: int = 5, 
    start_date: str="2024-06-28 12:00:00",
    n_eval_docs: int = 7,
    removal_condition: str = "loose", # strict or loose
    valid_models={"mallet", "ctm", "bertopic", "category-45"},
    valid_datasets={"wikitext-labeled", "bills-labeled"},
    path_save: str ="/Users/lbartolome/Documents/GitHub/theta-evaluation/data/files_pilot"
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
    
    ##############################
    # Filters for second round 
    ##############################
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

        # strict removal
        r["remove"] = (
            r["failed_purpose"]
            or r["failed_category"]
            or r["too_quick"]
            or r["failed_fit_check"]
            or r["failed_sponge_check_weak"]
            or r["failed_practice_rank_weak"]
            or (
                r["failed_fam_check"]
                and (r["failed_practice_rank_strict"] or r["failed_sponge_check_strict"])
            )
        )
        
        # loose removal
        failure_count = (
            r["failed_sponge_check_weak"] * 2
            + r["failed_fit_check"] * 2
            + r["failed_category"] * 2
            + r["too_quick"]
            + r["failed_purpose"]
            + r["failed_fam_check"]
            + r["failed_practice_rank_weak"]
        )
        r["remove_loose"] = failure_count > 1
                    
        r["StartDate"] = row["StartDate"]
        r["time"] = float(row["Duration (in seconds)"])
        r["eval_docs"] = deepcopy(cluster_data["eval_docs"])
        r["exemplar_docs"] = cluster_data["exemplar_docs"]
        r["topic_words"] = cluster_data["topic_words"]
        r["topic_match_id"] = cluster_data["topic_match_id"]
        
        label = row["cluster_label"]
        clarity = int(row["cluster_coherence"].split("-")[0].strip())
        
        # Normalize the keys

        for i in range(n_eval_docs):
            fit_answer = int(row[f"{i+1}_loop_fit_a"].split("-")[0].strip())
            r["eval_docs"][i]["fit"] = fit_answer
            r["eval_docs"][i]["rank"] = int(row[f"rank_{i}"])
            r["eval_docs"][i]["is_familiar"] = not str(row[f"{i+1}_loop_familiarity"]).startswith("I am not familiar")
                    
        responses.append(r)
    
    # Filter out disqualified responses
    print(f"Total responses: {len(responses)}")
    if removal_condition == "strict":
        col_remove = "remove"
    elif removal_condition == "loose":
        col_remove = "remove_loose"
    else:
        raise ValueError(f"Invalid removal condition: {removal_condition}")
    
    print(f"Removed: {sum(r[col_remove] for r in responses)}")
    responses = [r for r in responses if not r[col_remove]]
    responses_by_id = {cluster_id: [] for cluster_id in eval_data.keys()}
    for r in responses:
        responses_by_id[r["cluster_id"]].append(r)
        
    # normalize the "cluster_id" keys
    responses_by_id = {normalize_key(key, valid_models, valid_datasets): value for key, value in responses_by_id.items()}
            
    # Save cluster rank counts
    counts = Counter([r["cluster_id"] for r in responses])
    if path_save:
        with open(f"{path_save}/cluster_rank_counts.json", "w") as outfile:
            json.dump(counts, outfile, indent=2)
            
    return responses_by_id


def compute_agreement_per_topic(
    responses_by_id: dict[str, List[dict]],
    fit_threshold: int = 4,
    n_eval_docs: int = 7, 
):
    """
    Compute agreement metrics (Krippendorff's alpha and Gwet's AC2 for fit and rank data) for each topic.
    """
    agreement_data = []
    #bin_fit_data_by_model = {"mallet": [], "ctm": [], "category-45": []}
    bin_fit_data_by_model = defaultdict(list)

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
    model_fit_data = defaultdict(list)
    model_rank_data = defaultdict(list)
    model_prob_data = defaultdict(list)
    
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

    # Sort correlation data by topic ID
    corr_data = sorted(corr_data, key=lambda x: x["id"])
    
    return model_fit_data, model_rank_data, model_prob_data, corr_data

def compute_correlations_one(
    corr_data,
    rank_llm_data=None,
    fit_llm_data=None,
    aggregation_method="mean",
    fit_threshold=4,
    rescale_ndcg=True,
):
    """Compute correlation coefficients for fit and rank data: average rank per question/document over annotators, then correlate those with the topic model probabilities (thetas).
    """
    
    if rank_llm_data is not None:
        assert [k['id'] for k in corr_data] == [k['id'] for k in rank_llm_data], \
            "rank_llm_data does not have the same 'id' keys as corr_data"
    else:
        rank_llm_data = [None] * len(corr_data)
    
    if fit_llm_data is not None:
        assert [k['id'] for k in corr_data] == [k['id'] for k in fit_llm_data], \
            "fit_llm_data does not have the same 'id' keys as corr_data"
    else:
        fit_llm_data = [None] * len(corr_data)

    ndcg_score_ = ndcg_score
    if rescale_ndcg:
        n_items = corr_data[0]["rank_data"].shape[-1]
        rs = range(n_items)
        min_ndcg = ndcg_score([rs], [rs[::-1]])
        # rescale to be between 0 and 1
        ndcg_score_ = lambda x, y: (ndcg_score(x, y) - min_ndcg) / (1 - min_ndcg)

    corr_results = []
    for d_us, d_r_llm, d_f_llm in zip(corr_data, rank_llm_data, fit_llm_data):
        fit_data = d_us["fit_data"]
        rank_data = d_us["rank_data"]
        prob_data = d_us["prob_data"]
        assign_data = d_us["assign_data"]
        
        annotator_results = {} # to store llm-correlations
        f_llm, r_llm = None, None
        
        if d_r_llm is not None:
            r_llm = d_r_llm["rank_data"]
            r_annotators = d_r_llm["annotators"]
        
        if d_f_llm is not None:
            f_llm = d_f_llm["fit_data"]
            f_annotators = d_f_llm["annotators"] # f_annotators should be the same as r_annotators unless only one out of rank_llm_data and fit_llm_data is provided
        if aggregation_method == "mean":
            mean_fit_data, mean_rank_data = fit_data.mean(0), rank_data.mean(0)
        elif aggregation_method == "concatenate":
            mean_fit_data, mean_rank_data = fit_data.flatten(), rank_data.flatten()
            prob_data = np.tile(prob_data, len(fit_data))
        elif aggregation_method == "kemeny_young":
            n_items = fit_data.shape[1]
            preferences = np.zeros((n_items, n_items))
            for i, j in itertools.combinations(range(n_items), 2):
                preferences[i,j] = np.sum(rank_data[:, i] < rank_data[:, j])
                preferences[j,i] = np.sum(rank_data[:, i] > rank_data[:, j])
            
            best_score = float('-inf')
            best_ranking = None
            for perm in itertools.permutations(range(n_items)):
                score = sum(preferences[i,j] for i, j in itertools.combinations(perm, 2))
                if score > best_score:
                    best_score = score
                    best_ranking = perm
            
            mean_rank_data = np.array([i + 1 for i in best_ranking])
            mean_fit_data = fit_data.mean(0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Compute general correlations
        fit_rho, _ = spearmanr(mean_fit_data, prob_data)
        rank_rho, _ = spearmanr(mean_rank_data, prob_data)
        fit_tau, _ = kendalltau(mean_fit_data, prob_data)
        rank_tau, _ = kendalltau(mean_rank_data, prob_data)
        fit_ndcg = ndcg_score_([mean_fit_data], [prob_data])
        rank_ndcg = ndcg_score_([mean_rank_data], [prob_data])
        
        # Calculate fit agreement
        # binarized_fit_data = (fit_data >= fit_threshold).astype(int) # alternative to _mv
        binarized_fit_mv = (np.round(np.mean(fit_data, axis=0)) >= fit_threshold).astype(int)

        fit_agree = np.mean(binarized_fit_mv == assign_data)
        
        # LLM correlations if r_llm
        if r_llm is not None:
            for a, r_llm_a in zip(r_annotators, r_llm):
                rank_rho_users, _ = spearmanr(mean_rank_data, r_llm_a)
                rank_rho_gt, _ = spearmanr(prob_data, r_llm_a)
                rank_tau_users, _ = kendalltau(mean_rank_data, r_llm_a)
                rank_tau_gt, _ = kendalltau(prob_data, r_llm_a)
                rank_ndcg_users = ndcg_score_([mean_rank_data], [r_llm_a])
                rank_ndcg_gt = ndcg_score_([r_llm_a], [prob_data])
                
                # Add annotator-specific results to the dictionary
                annotator_results[f"rank_rho_users_{a}"] = rank_rho_users
                annotator_results[f"rank_rho_tm_{a}"] = rank_rho_gt
                annotator_results[f"rank_tau_users_{a}"] = rank_tau_users
                annotator_results[f"rank_tau_tm_{a}"] = rank_tau_gt
                annotator_results[f"rank_ndcg_users_{a}"] = rank_ndcg_users
                annotator_results[f"rank_ndcg_tm_{a}"] = rank_ndcg_gt
        
        # LLM correlations if f_llm
        if f_llm is not None:
            
            for a, f_llm_a in zip(f_annotators, f_llm):

                # Add annotator-specific results to the dictionary
                annotator_results[f"fit_tau_users_{a}"] = kendalltau(mean_fit_data, f_llm_a)[0]
                annotator_results[f"fit_tau_tm_{a}"] = kendalltau(prob_data, f_llm_a)[0]
                annotator_results[f"fit_agree_users_{a}"] = np.mean(binarized_fit_mv == f_llm_a)
                annotator_results[f"fit_agree_tm_{a}"] = np.mean(assign_data == f_llm_a)


        corr_results.append({
            "id": d_us["id"],
            "model": d_us["model"],
            "topic": d_us["topic"],
            "n_annotators": d_us["n_annotators"],
            "fit_rho": fit_rho,
            "fit_tau": fit_tau,
            "rank_rho": rank_rho,
            "rank_tau": rank_tau,
            "fit_agree": fit_agree,
            "fit_ndcg": fit_ndcg,
            "rank_ndcg": rank_ndcg,
            **annotator_results 
        })
    
    return pd.DataFrame(corr_results)

def compute_correlations_two(responses_by_id, rank_llm_data=None, fit_llm_data=None, fit_threshold = 4):
    
    corr_data = []    
    model_fit_data = defaultdict(list)
    model_rank_data = defaultdict(list)
    model_prob_data = defaultdict(list)

    # first collect the responses and compute the correlation data
    for id, group in responses_by_id.items():
        # get data from the id
        split_id = id.split("/")
        model_name = split_id[-2]
        topic = split_id[-1]

        if len(group) < 2:
            continue

        # compile the fit and rank data
        fit_data = np.array([
            [doc["fit"] for doc in r["eval_docs"]]
            for r in group
        ])
        rank_data = np.array([
            [doc["rank"] for doc in r["eval_docs"]]
            for r in group
        ])
        rank_data = 8 - rank_data # reverse so that higher is better
        prob_data = np.array(
            [doc["prob"] for doc in group[0]["eval_docs"]]
        )
        assign_data = np.array(
            [doc["assigned_to_k"] for doc in group[0]["eval_docs"]]
        )
        top_words = " ".join(group[0]["topic_words"][:10])
        avg_familiar = np.mean([doc["is_familiar"] for r in group for doc in r["eval_docs"]])

        model_fit_data[model_name].append(fit_data)
        model_rank_data[model_name].append(rank_data)
        model_prob_data[model_name].append(np.repeat([prob_data], len(group), axis=0))
        
        bin_fit_data = (fit_data >= fit_threshold).astype(int)
        
        for i in range(len(group)): # for each of the human annotators
            
            # compute the correlations
            fit_rho, _ = spearmanr(fit_data[i], prob_data)
            rank_rho, _ = spearmanr(rank_data[i], prob_data)

            fit_tau, _ = kendalltau(fit_data[i], prob_data)
            rank_tau, _ = kendalltau(rank_data[i], prob_data)

            fit_ndcg = ndcg_score([fit_data[i]], [prob_data])
            rank_ndcg = ndcg_score([rank_data[i]], [prob_data])

            fit_agree = np.mean(bin_fit_data[i] == assign_data)

            # to mean of other annotators
            other_mean_fit_data = np.mean(np.delete(fit_data, i, axis=0), axis=0)
            other_mean_rank_data = np.mean(np.delete(rank_data, i, axis=0), axis=0)

            fit_ia_rho, _ = spearmanr(fit_data[i], other_mean_fit_data)
            rank_ia_rho, _ = spearmanr(rank_data[i], other_mean_rank_data)

            fit_ia_tau, _ = kendalltau(fit_data[i], other_mean_fit_data)
            rank_ia_tau, _ = kendalltau(rank_data[i], other_mean_rank_data)            
            
            # if there is LLM data, compute the correlations
            annotator_results = {} 
            if rank_llm_data is not None:
                # keep element of the list rank_llm_data whose "id" == id
                rank_llm_group = list(filter(lambda d: d["id"] == id, rank_llm_data))[0]
                
                for llm_annotator, rank_llm in zip(rank_llm_group["annotators"], rank_llm_group["rank_data"]):
                    
                    # compute the correlations
                    rank_rho_user_llm, _ = spearmanr(rank_data[i], rank_llm)
                    rank_rho_llm_tm, _ = spearmanr(rank_llm, prob_data)
                    
                    rank_tau_user_llm, _ = kendalltau(rank_data[i], rank_llm)
                    rank_tau_llm_tm, _ = kendalltau(rank_llm, prob_data)

                    rank_ndcg_user_llm = ndcg_score([rank_data[i]], [rank_llm])
                    rank_ndcg_llm_tm = ndcg_score([rank_llm], [prob_data])
                    
                    # add to the dictionary
                    annotator_results[f"rank_rho_user_{llm_annotator}"] = rank_rho_user_llm
                    annotator_results[f"rank_rho_tm_{llm_annotator}"] = rank_rho_llm_tm
                    annotator_results[f"rank_tau_user_{llm_annotator}"] = rank_tau_user_llm
                    annotator_results[f"rank_tau_tm_{llm_annotator}"] = rank_tau_llm_tm
                    annotator_results[f"rank_ndcg_user_{llm_annotator}"] = rank_ndcg_user_llm
                    annotator_results[f"rank_ndcg_tm_{llm_annotator}"] = rank_ndcg_llm_tm
                
            if fit_llm_data is not None:
                fit_llm_group = list(filter(lambda d: d["id"] == id, fit_llm_data))[0]
                
                for llm_annotator, fit_llm in zip(fit_llm_group["annotators"], fit_llm_group["fit_data"]):
                    fit_rho_user_llm, _ = spearmanr(fit_data[i], fit_llm)
                    fit_rho_llm_tm, _ = spearmanr(fit_llm, prob_data)
                    
                    fit_tau_user_llm, _ = kendalltau(fit_data[i], fit_llm)
                    fit_tau_llm_tm, _ = kendalltau(fit_llm, prob_data)
                    
                    fit_ndcg_user_llm = ndcg_score([fit_data[i]], [fit_llm])
                    fit_ndcg_llm_tm = ndcg_score([fit_llm], [prob_data])
                    
                    # add to the dictionary
                    annotator_results[f"fit_rho_user_{llm_annotator}"] = fit_rho_user_llm
                    annotator_results[f"fit_rho_tm_{llm_annotator}"] = fit_rho_llm_tm
                    annotator_results[f"fit_tau_user_{llm_annotator}"] = fit_tau_user_llm
                    annotator_results[f"fit_tau_tm_{llm_annotator}"] = fit_tau_llm_tm
                    annotator_results[f"fit_ndcg_user_{llm_annotator}"] = fit_ndcg_user_llm
                    annotator_results[f"fit_ndcg_tm_{llm_annotator}"] = fit_ndcg_llm_tm
            
            corr_data.append({
                "id": id,
                "model": model_name,
                "n_annotators": len(group),
                "topic": topic,
                "topic_match_id": group[0]["topic_match_id"],
                "annotator": i,
                "fit_rho": fit_rho,
                "fit_tau": fit_tau,
                "rank_rho": rank_rho,
                "rank_tau": rank_tau,
                "fit_NDCG": fit_ndcg,
                "rank_NDCG": rank_ndcg,
                "fit_agree": fit_agree,
                #"rank_agree": rank_agree,
                "fit_ia-rho": fit_ia_rho,
                "fit_ia-tau": fit_ia_tau,
                "rank_ia-rho": rank_ia_rho,
                "rank_ia-tau": rank_ia_tau,
                **annotator_results
            })
            
    corr_data = pd.DataFrame(corr_data)
    return corr_data

def calculate_coherence(config_pilot, data_docs=[
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/files_pilot/document-data/wikitext/processed/labeled/vocab_15k/train.metadata.jsonl",
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/files_pilot/document-data/wikitext/processed/labeled/vocab_15k/test.metadata.jsonl"
]):
    """Calculate NPMI coherence for each topic in eval_data."""
    docs = []
    for data_doc in data_docs:
        with open(data_doc) as infile:
            docs.extend([json.loads(line)["tokenized_text"].split() for line in infile])
    
    vocab = sorted(set([word for doc in docs for word in doc]))
    data_dict = Dictionary([vocab])
    
    npmi_data = []
    for model_id, model_data in config_pilot.items(): # this selects the set of topics for a given model (mallet, ctm, etc.)
        for cluster_id, cluster_data in model_data.items(): # each cluster_data is the information for a topic
            id_ = f"{model_id}/{cluster_id}"
            topics = [cluster_data["topic_words"][:15]] # number shown to annotators
            cm = CoherenceModel(
                topics=topics,
                texts=docs,
                dictionary=data_dict,
                coherence="c_npmi",
                window_size=None,
                processes=1,
            )
            confirmed_measures = cm.get_coherence_per_topic()
            mean = cm.aggregate_measures(confirmed_measures)
            npmi_data.append({"id": id_, "NPMI": mean})
    
    npmi_data = pd.DataFrame(npmi_data)

    return npmi_data

def calculate_corr_npmi(corr_data, npmi_data, columns_calculate = ["rank_rho", "rank_rtau"]):
    
    corr_data = corr_data.merge(npmi_data, on="id")    
    corrs = []
    for column in columns_calculate:
        spearman_corr, spearman_p = spearmanr(corr_data[column], corr_data['NPMI'])
        kendall_corr, kendall_p = kendalltau(corr_data[column], corr_data['NPMI'])
        corrs.append({
            'Metric': column,
            'Spearman_rho': spearman_corr,
            'Spearman_p_value': spearman_p,
            'Kendall_tau': kendall_corr,
            'Kendall_p_value': kendall_p
        })
    corrs_df = pd.DataFrame(corrs)
    
    return corrs_df