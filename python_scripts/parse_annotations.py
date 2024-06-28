#%%
import json
from copy import deepcopy
from collections import Counter
from itertools import combinations, groupby

import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import ndcg_score


RESPONSE_CSV = "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_June+27%2C+2024_09.47.csv"
DATA_JSON = "../data/json_out/config_pilot.json"

raw_responses = pd.read_csv(RESPONSE_CSV)
MIN_MINUTES = 5
START_DATE = "2024-06-28 00:00:00"
n_eval_docs = 7

# %% load data
# first two rows are junk from Qualtrics
column_names = dict(zip(raw_responses.columns, raw_responses.iloc[0]))
raw_responses = raw_responses.iloc[2:]
# remove preview data
raw_responses = raw_responses.loc[raw_responses["Status"] == "IP Address"]
# only after recent change
raw_responses = raw_responses.loc[raw_responses["StartDate"] >= START_DATE]

# %% get the data
with open(DATA_JSON) as infile:
    eval_data = json.load(infile)
eval_data = {
    f"{model_id}/{cluster_id}": cluster_data
    for model_id, model_data in eval_data.items()
    for cluster_id, cluster_data in model_data.items()
}

# %% check that the columns are correct
# first for the loop-and-merge fit questions
assert column_names["11_loop_fit_a"].startswith("Attention Check") # should have been "7" but "11" due to qualtrics bug
assert column_names["1_loop_fit_a"].startswith("${e://Field/eval_doc_0}")
assert column_names[f"{n_eval_docs}_loop_fit_a"].startswith(f"${{e://Field/eval_doc_{n_eval_docs-1}}}")

# then for the ranking questions
assert column_names["rank_99"].endswith("distractor_doc") # attention check
assert column_names["rank_0"].endswith("eval_doc_0")
assert column_names[f"rank_{n_eval_docs-1}"].endswith(f"eval_doc_{n_eval_docs-1}")

assert "prolific ID" in column_names["Q22"]

# %% parse
responses = []
total_time = []
failed_fit_att_check, failed_rank_att_check, failed_practice_rank, too_quick = 0, 0, 0, 0

time_cutoff = min(np.quantile(raw_responses["Duration (in seconds)"][2:].astype(float), 0.05), 60 * MIN_MINUTES) # 4 minutes

for _, row in raw_responses.iterrows():
    r = {}
    if float(row["Duration (in seconds)"]) < time_cutoff:
        too_quick += 1
        continue

    if row["11_loop_fit_a"] != "Not sure":
        failed_fit_att_check += 1
        continue

    if row[f"rank_99"] not in ["8"]: # change to 7, 8?
        failed_rank_att_check += 1
        continue

    r["annotator_id"] = row["Q22"]
    r["time"] = float(row["Duration (in seconds)"])

    # retrieve the data for the cluster/topic that was used to generate the questions for this respondent
    r["cluster_id"] = row["id"]
    cluster_data = eval_data[r["cluster_id"]]

    r["eval_docs"] = deepcopy(cluster_data["eval_docs"])
    assert len(r["eval_docs"]) == n_eval_docs
    r["exemplar_docs"] = cluster_data["exemplar_docs"]
    r["topic_words"] = cluster_data["topic_words"]

    # did they do the practice ranking correctly?
    practice_ranks = [int(row[f"practice_rank_{i}"]) for i in range(4)]
    r["is_practice_rank_correct"] = practice_ranks == [1, 2, 3, 4]
    if not r["is_practice_rank_correct"]:
        failed_practice_rank += 1
        continue
    
    # get their assigned label and clarity score
    label = row["cluster_label"]
    clarity = int(row["cluster_coherence"].split("-")[0].strip())

    # now get their responses for each document
    for i in range(n_eval_docs):
        fit_answer = row[f"{i+1}_loop_fit_a"]
        if fit_answer == "No, it doesn't fit":
            fit_answer = 0
        elif fit_answer == "Not sure":
            fit_answer = 1
        elif fit_answer == "Yes, it fits":
            fit_answer = 2
        
        r["eval_docs"][i]["fit"] = fit_answer
        r["eval_docs"][i]["rank"] = int(row[f"rank_{i}"])

    responses.append(r)

print(f"Total responses: {len(responses)}")
print(f"Failed fit attention check: {failed_fit_att_check}")
print(f"Failed rank attention check: {failed_rank_att_check}")
print(f"Too quick: {too_quick}")
print(f"Failed practice ranking: {failed_practice_rank}")

responses = sorted(responses, key=lambda r: r["cluster_id"])

# %% Save the id counts
counts = Counter([r["cluster_id"] for r in responses])
with open("../data/human_annotations/_cluster_rank_counts.json", "w") as outfile:
    json.dump(counts, outfile, indent=2)

# %% Bonus calculation: if correlation between any two people is over threshold,
# both get a bonus
min_corr_agree = 0.5
bonus_receivers = set()
for id, group in groupby(responses, key=lambda r: r["cluster_id"]):
    group = list(group)
    if len(group) < 2:
        continue
    for r1, r2 in combinations(group, 2):
        r1_ranks = [doc["rank"] for doc in r1["eval_docs"]]
        r2_ranks = [doc["rank"] for doc in r2["eval_docs"]]
        corr, _ = spearmanr(r1_ranks, r2_ranks)
        if corr >= min_corr_agree:
            bonus_receivers.add(r1["annotator_id"])
            bonus_receivers.add(r2["annotator_id"])

print(f"# Bonus receivers: {len(bonus_receivers)}")
print("\n".join(sorted(bonus_receivers)))

# %% A summary of the responses
for i, r in enumerate(responses):
    fits = np.array([doc["fit"] for doc in r["eval_docs"]])
    ranks = np.array([doc["rank"] for doc in r["eval_docs"]])
    probs = np.array([doc["prob"] for doc in r["eval_docs"]])
    assigns = np.array([doc["assigned_to_k"] for doc in r["eval_docs"]])

    # fit-to-rank correlation
    fit_to_rank_corr, fit_to_rank_pval = spearmanr(fits, ranks)

    # fit to prob correlation
    fit_to_prob_corr, fit_to_prob_pval = spearmanr(fits, probs)

    # rank to prob correlation
    rank_to_prob_corr, rank_to_prob_pval = spearmanr(ranks, probs)

    print(f"\n ==== Cluster {r['cluster_id']}, annotator {i}, time {r['time']/60:0.1f} ====")
    print(
        f"Fit to rank corr: {fit_to_rank_corr:.2f} (p={fit_to_rank_pval:.3f}) | "
        f"Fit to prob corr: {fit_to_prob_corr:.2f} (p={fit_to_prob_pval:.3f}) | "
        f"Rank to prob corr: {rank_to_prob_corr:.2f} (p={rank_to_prob_pval:.3f}) | "
    )
# %%
from itertools import groupby
from nltk.metrics.agreement import AnnotationTask
from krippendorff import alpha

for id, group in groupby(responses, key=lambda r: r["cluster_id"]):
    group = list(group)
    if len(group) < 2:
        continue
    fit_data = np.array([
        [r["eval_docs"][i]["fit"] for i in range(n_eval_docs)]
        for r in group
    ])
    alpha_score = alpha(fit_data, value_domain=[0, 1, 2])
    print (f"Cluster {id}, fit agreement ({len(group)}): {alpha_score:0.3f}")

    rank_data = np.array([
        [r["eval_docs"][i]["rank"] for i in range(n_eval_docs)]
        for r in group
    ])
    alpha_score = alpha(rank_data, value_domain=range(1, n_eval_docs+1), level_of_measurement="ordinal")
    print (f"Cluster {id}, rank agreement ({len(group)}): {alpha_score:0.3f}")

    # # get the fits
    # fit_task = AnnotationTask(data=[
    #     (r["annotator_id"], i, r["eval_docs"][i]["fit"])
    #     for r in group
    #     for i in range(n_eval_docs)
    # ])
    # print(f"Cluster {id}, fit agreement ({len(group)}): {fit_task.alpha():0.3f}")
    
    # # get the ranks
    # rank_task = AnnotationTask(data=[
    #     (r["annotator_id"], i, r["eval_docs"][i]["rank"])
    #     for r in group
    #     for i in range(n_eval_docs)
    # ])
    # print(f"Cluster {id}, rank agreement ({len(group)}): {rank_task.alpha():0.3f}")

# %%
