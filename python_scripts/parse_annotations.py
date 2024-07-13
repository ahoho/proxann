#%%
import json
from copy import deepcopy
from collections import Counter
from itertools import groupby

import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import ndcg_score


RESPONSE_CSV = "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_June+29%2C+2024_15.48.csv"
DATA_JSON = "../data/json_out/config_first_round.json"

raw_responses = pd.read_csv(RESPONSE_CSV)
MIN_MINUTES = 5
START_DATE = "2024-06-28 12:00:00" # TODO: possibly start from 06-29 given some changes
n_eval_docs = 7

# %% load data
# first two rows are junk from Qualtrics
column_names = dict(zip(raw_responses.columns, raw_responses.iloc[0]))
raw_responses = raw_responses.iloc[2:]
# remove preview data
raw_responses = raw_responses.loc[raw_responses["Status"] == "IP Address"]
# only after recent change
raw_responses = raw_responses.loc[raw_responses["StartDate"] >= START_DATE]

print(f"Total responses: {len(raw_responses)}")

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

time_cutoff = min(np.quantile(raw_responses["Duration (in seconds)"][2:].astype(float), 0.05), 60 * MIN_MINUTES) # 4 minutes

for _, row in raw_responses.iterrows():
    r = {}
    r["annotator_id"] = row["Q22"]

    # check if completion time was too fast
    r["too_quick"] = float(row["Duration (in seconds)"]) < time_cutoff

    # check basic comprehension
    r["failed_purpose"] = "single category" not in row["practice_purpose"]

    # check attention checks
    r["failed_fit_check"] = not row["11_loop_fit_a"].startswith("2")
    r["failed_fam_check"] = not str(row["11_loop_familiarity"]).startswith("I am not familiar")

    r["failed_sponge_check_strict"] = row[f"rank_99"] != "8"
    r["failed_sponge_check_weak"] = row[f"rank_99"] not in ["7", "8"]

    # did they do the practice ranking correctly?
    practice_ranks = [int(row[f"practice_rank_{i}"]) for i in range(4)]
    r["failed_practice_rank_strict"] = practice_ranks != [1, 2, 3, 4]
    # extremely obvious
    r["failed_practice_rank_weak"] = practice_ranks[0] not in [1,2] or practice_ranks[3] != 4

    # determine when to throw people away
    r["remove"] = (
        r["failed_purpose"] # if they got this wrong they almost certainly didn't understand the task
        or r["too_quick"] # much too quick and they probably didn't read the questions
        or r["failed_fit_check"] # this attention check is very clear
        or r["failed_sponge_check_weak"] # if you didn't put it anywhere close to last, you didn't follow
        or r["failed_practice_rank_weak"] # if you didn't put the first and last in the right place, you didn't follow
        or (
            # missing familiarity checkbox is OK, but if you get something else slightly wrong, you're out
            r["failed_fam_check"] and (r["failed_practice_rank_strict"] or r["failed_sponge_check_strict"])
        )
    )

    r["StartDate"] = row["StartDate"]
    r["time"] = float(row["Duration (in seconds)"])

    # retrieve the data for the cluster/topic that was used to generate the questions for this respondent
    r["cluster_id"] = row["id"]
    cluster_data = eval_data[r["cluster_id"]]

    r["eval_docs"] = deepcopy(cluster_data["eval_docs"])
    assert len(r["eval_docs"]) == n_eval_docs
    r["exemplar_docs"] = cluster_data["exemplar_docs"]
    r["topic_words"] = cluster_data["topic_words"]
    
    # get their assigned label and clarity score
    label = row["cluster_label"]
    clarity = int(row["cluster_coherence"].split("-")[0].strip())

    # now get their responses for each document
    for i in range(n_eval_docs):
        fit_answer = int(row[f"{i+1}_loop_fit_a"].split("-")[0].strip())
        
        r["eval_docs"][i]["fit"] = fit_answer
        r["eval_docs"][i]["rank"] = int(row[f"rank_{i}"])
        r["eval_docs"][i]["is_familiar"] = not str(row[f"{i+1}_loop_familiarity"]).startswith("I am not familiar")

    responses.append(r)

#%% Collect the attention check failures
print(f"Total responses: {len(responses)}")
print(f"Removed: {sum(r['remove'] for r in responses)}")
print(f"Too quick: {sum(r['too_quick'] for r in responses)}")
print(f"Failed purpose: {sum(r['failed_purpose'] for r in responses)}")
print(f"Failed fit check: {sum(r['failed_fit_check'] for r in responses)}")
print(f"Failed fam check: {sum(r['failed_fam_check'] for r in responses)}")
print(f"Failed sponge check weak: {sum(r['failed_sponge_check_weak'] for r in responses)}")
print(f"Failed sponge check strict: {sum(r['failed_sponge_check_strict'] for r in responses)}")
print(f"Failed practice rank weak: {sum(r['failed_practice_rank_weak'] for r in responses)}")
print(f"Failed practice rank strict: {sum(r['failed_practice_rank_strict'] for r in responses)}")

responses = [r for r in responses if not r["remove"]]
responses = sorted(responses, key=lambda r: r["cluster_id"])

# %% Save the id counts
counts = Counter([r["cluster_id"] for r in responses])
with open("../data/human_annotations/_cluster_rank_counts.json", "w") as outfile:
    json.dump(counts, outfile, indent=2)

# %% Save the responses
# get date from the last response
most_recent_date = sorted(responses, key=lambda r: r["StartDate"])[-1]["StartDate"]
most_recent_date = most_recent_date.replace("-", "").replace(":", "").replace(" ", "_")
with open(f"../data/human_annotations/{most_recent_date}_parsed_responses.json", "w") as outfile:
    json.dump(responses, outfile, indent=2)

# %% Bonus calculation: if correlation between an annotator and the average
# is above a certain threshold, they get a bonus

# first, get the list of people who have already been paid
with open("../data/human_annotations/paid_bonuses.txt") as infile:
    paid_bonuses = set(line.strip() for line in infile)

min_corr_agree = 0.75
payout=1.5
bonus_receivers = set()
for id, group in groupby(responses, key=lambda r: r["cluster_id"]):
    group = list(group)
    if len(group) < 3:
        continue
    ranks = np.array([
        [doc["rank"] for doc in r["eval_docs"]]
        for r in group
    ])
    for i, rank_i in enumerate(ranks):
        mean_rank_without_i = np.mean(np.delete(ranks, i, axis=0), axis=0)
        corr, _ = spearmanr(rank_i, mean_rank_without_i)
        if corr >= min_corr_agree:
            if not group[i]["failed_fam_check"]:
                bonus_receivers.add(group[i]["annotator_id"])

# if time spent above a certain threshold, they get a bonus
bonus_time = 60 * 15
for r in responses:
    if r["time"] > bonus_time:
        if not r["failed_fam_check"]:
            bonus_receivers.add(r["annotator_id"])

print(f"# Bonus receivers: {len(bonus_receivers)}")
print("\n".join([f"{id},{payout}" for id in sorted(bonus_receivers) if id not in paid_bonuses]))

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

for id, group in groupby(responses, key=lambda r: r["cluster_id"]):
    group = list(group)
    if len(group) < 2:
        continue
    fit_data = np.array([
        [doc["fit"] for doc in r["eval_docs"]]
        for r in group
    ])
    rank_data = np.array([
        [doc["rank"] for doc in r["eval_docs"]]
        for r in group
    ])
    prob_data = np.array(
        [doc["prob"] for doc in group[0]["eval_docs"]]
    )
    print("")
    top_words = " ".join(group[0]["topic_words"][:10])
    avg_familiar = np.mean([doc["is_familiar"] for r in group for doc in r["eval_docs"]])
    print(f"=== {id} | annotators: {len(group)} | {top_words} | {avg_familiar:.2f} fam ===")
    # average inter-annotator correlations
    ia_fit_corrs = np.zeros((len(group), 2))
    ia_rank_corrs = np.zeros((len(group), 2))
    for i in range(len(group)):
        fit_i = fit_data[i]
        rank_i = rank_data[i]
        # is this leave-one-out averaging the right choice? should we sum instead?
        mean_fit = np.mean(np.delete(fit_data, i, axis=0), axis=0)
        mean_rank = np.mean(np.delete(rank_data, i, axis=0), axis=0)

        ia_fit_corrs[i] = spearmanr(fit_i, mean_fit)
        ia_rank_corrs[i] = spearmanr(rank_i, mean_rank)
    mean_ia_fit_corr = np.mean(ia_fit_corrs, axis=0)
    mean_ia_rank_corr = np.mean(ia_rank_corrs, axis=0)
    #print(f"mean IA fit correlation: {mean_ia_fit_corr[0]:0.3f} (p={mean_ia_fit_corr[1]:0.3f})")
    #print(f"mean IA rank correlation: {mean_ia_rank_corr[0]:0.3f} (p={mean_ia_rank_corr[1]:0.3f})")

    # average model-annotator correlations
    # TODO: definitely want to double check this--concatenate instead??
    prob_fit_corrs = np.zeros((len(group), 2))
    prob_rank_corrs = np.zeros((len(group), 2))
    for i in range(len(group)):
        fit_i = fit_data[i]
        rank_i = rank_data[i]
        prob_fit_corrs[i] = spearmanr(fit_i, prob_data)
        prob_rank_corrs[i] = spearmanr(rank_i, prob_data)
    mean_prob_fit_corr = np.mean(prob_fit_corrs, axis=0)
    mean_prob_rank_corr = np.mean(prob_rank_corrs, axis=0)

    # TODO: bootstrap (with leave-one-out or something)

    print(f"mean fit-prob correlation: {mean_prob_fit_corr[0]:0.3f} (p={mean_prob_fit_corr[1]:0.3f})")
    print(f"mean rank-prob correlation: {mean_prob_rank_corr[0]:0.3f} (p={mean_prob_rank_corr[1]:0.3f})")

    # Borda count
    rank_sums = (8 - rank_data).sum(0)
    rank_sum_corr, rank_sum_pval = spearmanr(rank_sums, prob_data)
    print(f"rank sum-prob correlation: {rank_sum_corr:0.3f} (p={rank_sum_pval:0.3f})")

    # agreements
    alpha_score = alpha(fit_data, value_domain=[1, 2, 3, 4, 5], level_of_measurement="ordinal")
    #print (f"fit agreement ({len(group)}): {alpha_score:0.3f}")

    alpha_score = alpha(rank_data, value_domain=range(1, n_eval_docs+2), level_of_measurement="ordinal")
    #print (f"rank agreement ({len(group)}): {alpha_score:0.3f}")
    
    # # get the fits
    # fit_task = AnnotationTask(data=[
    #     (r["annotator_id"], i, r["eval_docs"][i]["fit"])
    #     for r in group
    #     for i in range(n_eval_docs)
    # ])
    # print(f"fit agreement ({len(group)}): {fit_task.alpha():0.3f}")
    
    # # get the ranks
    # rank_task = AnnotationTask(data=[
    #     (r["annotator_id"], i, r["eval_docs"][i]["rank"])
    #     for r in group
    #     for i in range(n_eval_docs)
    # ])
    # print(f"rank agreement ({len(group)}): {rank_task.alpha():0.3f}")


# %% agreement per topic
from irrCAC.raw import CAC

fits_threshold = 4
agreement_data = []
bin_fit_data_by_model = {"mallet": [], "ctm": [], "category-45": []}
j = 0
for id, group in groupby(responses, key=lambda r: r["cluster_id"]):
    group = list(group)
    if len(group) < 2:
        continue

    # name the indices of for the dataframes
    ann_idxs = [f"{id}_ann_{i}" for i in range(len(group))]
    doc_idxs = [f"{id}_doc_{i}" for i in range(len(group[0]["eval_docs"]))]

    # collect fit data
    fit_data = pd.DataFrame(
        [[doc["fit"] for doc in r["eval_docs"]] for r in group],
        index=ann_idxs,
        columns=doc_idxs,
    ).T
    # binarize the fits
    bin_fit_data = id + "_" + (fit_data >= fits_threshold).astype(str)

    # get rank data
    rank_data = pd.DataFrame(
        [[doc["rank"] for doc in r["eval_docs"]] for r in group],
        index=ann_idxs,
        columns=doc_idxs,
    ).T

    # create the agreement data
    fit_cac = CAC(fit_data, weights="ordinal", categories=[1, 2, 3, 4, 5])
    bin_fit_cac = CAC(bin_fit_data, weights="identity", categories=[f"{id}_True", f"{id}_False"])
    rank_cac = CAC(rank_data, weights="ordinal", categories=list(range(1, n_eval_docs+1)))

    fit_alpha = fit_cac.krippendorff()["est"]
    fit_ac2 = fit_cac.gwet()["est"]

    bin_fit_alpha = bin_fit_cac.krippendorff()["est"]
    bin_fit_ac2 =  bin_fit_cac.gwet()["est"]

    rank_alpha = fit_cac.krippendorff()["est"]
    rank_ac2 = fit_cac.gwet()["est"]

    # get data from the id
    split_id = id.split("/")
    model_name = split_id[-2]
    topic = split_id[-1]

    agreement_data.append({
        "id": id,
        "model": model_name,
        "topic": topic,

        "fit_alpha": fit_alpha["coefficient_value"],
        "fit_alpha_p": fit_alpha["p_value"],

        "fit_ac2": fit_ac2["coefficient_value"],
        "fit_ac2_p": fit_ac2["p_value"],

        "bin_fit_alpha": bin_fit_alpha["coefficient_value"],
        "bin_fit_alpha_p": bin_fit_alpha["p_value"],

        "bin_fit_ac2": bin_fit_ac2["coefficient_value"],
        "bin_fit_ac2_p": bin_fit_ac2["p_value"],

        "rank_alpha": rank_alpha["coefficient_value"],
        "rank_alpha_p": rank_alpha["p_value"],

        "rank_ac2": rank_ac2["coefficient_value"],
        "rank_ac2_p": rank_ac2["p_value"],
    })
    bin_fit_data_by_model[model_name].append(bin_fit_data)

agreement_data_by_topic = pd.DataFrame(agreement_data)

bin_agreement_data_by_model = []
for model, model_bin_fit_data in bin_fit_data_by_model.items():
    model_bin_fit_data = pd.concat(model_bin_fit_data)
    model_bin_fit_cac = CAC(model_bin_fit_data, weights="identity")

    model_bin_fit_alpha = model_bin_fit_cac.krippendorff()["est"]
    model_bin_fit_ac2 = model_bin_fit_cac.gwet()["est"]

    bin_agreement_data_by_model.append({
        "model": model,
        "bin_fit_alpha": model_bin_fit_alpha["coefficient_value"],
        "bin_fit_alpha_p": model_bin_fit_alpha["p_value"],

        "bin_fit_ac2": model_bin_fit_ac2["coefficient_value"],
        "bin_fit_ac2_p": model_bin_fit_ac2["p_value"],
    })

bin_agreement_data_by_model = pd.DataFrame(bin_agreement_data_by_model)

# %%
