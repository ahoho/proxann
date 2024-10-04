#%%
import json
from copy import deepcopy
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind, sem, mannwhitneyu
from sklearn.metrics import ndcg_score

from irrCAC.raw import CAC

from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_bar,
    geom_boxplot,
    geom_errorbar,
    facet_wrap,
    facet_grid,
    position_dodge,
    theme_classic,
    coord_cartesian,
    scale_fill_brewer,
    scale_fill_manual,
    scale_color_brewer,
    theme,
    element_text,
    element_blank,
)

#%%
RESPONSE_CSV = "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_July+14%2C+2024_15.13.csv"
DATA_JSONS = ["../data/json_out/config_first_round.json", "../data/json_out/config_second_round.json"]

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
eval_data = {}
topics_per_model = Counter()
for json_fpath in DATA_JSONS:
    with open(json_fpath) as infile:
        raw_eval_data = json.load(infile)
    
    for model_id, model_data in raw_eval_data.items():
        for cluster_id, cluster_data in model_data.items():
            # the match id should align the topics across models
            cluster_data["topic_match_id"] = topics_per_model[model_id]
            eval_data[f"{model_id}/{cluster_id}"] = cluster_data
            topics_per_model[model_id] += 1


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

    # don't include people who didn't complete
    if str(row["rank_99"]) == "nan":
        continue

    # retrieve the data for the cluster/topic that was used to generate the questions for this respondent
    r["cluster_id"] = row["id"]
    r["annotator_id"] = row["Q22"]
    cluster_data = eval_data[r["cluster_id"]]
    # check if completion time was too fast
    r["too_quick"] = float(row["Duration (in seconds)"]) < time_cutoff

    # store their label for the topic
    r["category"] = row["cluster_label"]

    # make sure they didn't just copy the topic
    r["failed_category"] = r["category"] in " ".join(cluster_data["topic_words"])

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
        or r["failed_category"] # if they just copied the topic, they are lazy
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
print(f"Failed category: {sum(r['failed_category'] for r in responses)}")
print(f"Failed purpose: {sum(r['failed_purpose'] for r in responses)}")
print(f"Failed fit check: {sum(r['failed_fit_check'] for r in responses)}")
print(f"Failed fam check: {sum(r['failed_fam_check'] for r in responses)}")
print(f"Failed sponge check weak: {sum(r['failed_sponge_check_weak'] for r in responses)}")
print(f"Failed sponge check strict: {sum(r['failed_sponge_check_strict'] for r in responses)}")
print(f"Failed practice rank weak: {sum(r['failed_practice_rank_weak'] for r in responses)}")
print(f"Failed practice rank strict: {sum(r['failed_practice_rank_strict'] for r in responses)}")

responses = [r for r in responses if not r["remove"]]
responses_by_id = {cluster_id: [] for cluster_id in eval_data.keys()}
for r in responses:
    responses_by_id[r["cluster_id"]].append(r)

# %% Save the id counts
counts = Counter([r["cluster_id"] for r in responses])
with open("../data/human_annotations/_cluster_rank_counts.json", "w") as outfile:
    json.dump(counts, outfile, indent=2)
print(json.dumps(counts, indent=1))

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
for id, group in responses_by_id.items():
    
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

#%% 
# collect the categories per topic -- this is just for a latex table, so doesn't need to be complete
category_data = []

for i, (id, group) in enumerate(responses_by_id.items()):
    if i % 3 != 0: # ensures matches
        continue
    if len(group) < 2:
        continue
    
    split_id = id.split("/")
    model_name = split_id[-2]
    topic = split_id[-1]

    category_data.append({
        "id": id,
        "model": model_name,
        "topic": topic,
        "topic_idx": i // 9,
        "annotator_id": None,
        "annotator_idx": -1,
        "category": r"\emph{" + " ".join(group[0]["topic_words"][:5]) + "}",
    })
    for j in range(4):
        try:
            category = group[j]["category"]
            annotator_id = group[j]["annotator_id"]
        except IndexError:
            category = ""
            annotator_id = None
        category_data.append({
            "id": id,
            "model": model_name,
            "topic": topic,
            "topic_idx": i // 9,
            "annotator_id": annotator_id,
            "annotator_idx": j,
            "category": category.replace("&", r"\&"),
        })

# create a simple dataframe for latex
category_data = (
    pd.DataFrame(category_data)
      .pivot_table(columns="model", index=["topic_idx", "annotator_idx"], values="category", aggfunc="first")
      #.droplevel(1, 0)
      [["mallet", "ctm", "category-45"]] # sort columns
      .rename(columns={"mallet": "Mallet", "ctm": "CTM", "category-45": "Labeled"})
)

# print the latex
# 
print(
    category_data.style.hide().to_latex(
        hrules=True
    )
)

# %% agreement per topic
fit_threshold = 4
agreement_data = []
bin_fit_data_by_model = {"mallet": [], "ctm": [], "category-45": []}
j = 0
for id, group in responses_by_id.items():
    
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
    bin_fit_data = id + "_" + (fit_data >= fit_threshold).astype(str)

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

    # bin_fit_alpha = bin_fit_cac.krippendorff()["est"]
    # bin_fit_ac2 =  bin_fit_cac.gwet()["est"]

    rank_alpha = rank_cac.krippendorff()["est"]
    rank_ac2 = rank_cac.gwet()["est"]

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

        # "bin_fit_alpha": bin_fit_alpha["coefficient_value"],
        # "bin_fit_alpha_p": bin_fit_alpha["p_value"],

        # "bin_fit_ac2": bin_fit_ac2["coefficient_value"],
        # "bin_fit_ac2_p": bin_fit_ac2["p_value"],

        "rank_alpha": rank_alpha["coefficient_value"],
        "rank_alpha_p": rank_alpha["p_value"],

        "rank_ac2": rank_ac2["coefficient_value"],
        "rank_ac2_p": rank_ac2["p_value"],
    })
    bin_fit_data_by_model[model_name].append(bin_fit_data)

agreement_data_by_topic = pd.DataFrame(agreement_data)

# bin_agreement_data_by_model = []
# for model, model_bin_fit_data in bin_fit_data_by_model.items():
#     model_bin_fit_data = pd.concat(model_bin_fit_data)
#     model_bin_fit_cac = CAC(model_bin_fit_data, weights="identity")

#     model_bin_fit_alpha = model_bin_fit_cac.krippendorff()["est"]
#     model_bin_fit_ac2 = model_bin_fit_cac.gwet()["est"]

#     bin_agreement_data_by_model.append({
#         "model": model,
#         "bin_fit_alpha": model_bin_fit_alpha["coefficient_value"],
#         "bin_fit_alpha_p": model_bin_fit_alpha["p_value"],

#         "bin_fit_ac2": model_bin_fit_ac2["coefficient_value"],
#         "bin_fit_ac2_p": model_bin_fit_ac2["p_value"],
#     })

# bin_agreement_data_by_model = pd.DataFrame(bin_agreement_data_by_model)
#%%
print(
    agreement_data_by_topic
        .replace({"mallet": "Mallet", "ctm": "CTM", "category-45": "Labeled"})
        .groupby("model")[["fit_alpha", "rank_alpha"]]
        .agg(lambda x: f"{x.mean():0.2f} ({x.std():0.2f})")
        .rename(columns={"fit_alpha": r"$\alpha$ Fit", "rank_alpha": r"$\alpha$ Rank"})
        .style
        .to_latex(hrules=True)
        .replace("model", "")
)

# %%
aggregation_method = "mean"
corr_data = []
model_fit_data = {"mallet": [], "ctm": [], "category-45": []}
model_rank_data = {"mallet": [], "ctm": [], "category-45": []}
model_prob_data = {"mallet": [], "ctm": [], "category-45": []}

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
    #print(f"=== {id} | annotators: {len(group)} | {top_words} | {avg_familiar:.2f} fam ===")

    model_fit_data[model_name].append(fit_data)
    model_rank_data[model_name].append(rank_data)
    model_prob_data[model_name].append(np.repeat([prob_data], len(group), axis=0))
    
    # compute the correlations
    bin_fit_data = (fit_data >= fit_threshold).astype(int)
    # could flatten and compute over all rather than take mean
    if aggregation_method == "mean":
        mean_rank_data = rank_data.mean(0)
        mean_fit_data = fit_data.mean(0)
    elif aggregation_method == "concatenate":
        mean_fit_data = fit_data.flatten()
        mean_rank_data = rank_data.flatten()
        prob_data = np.tile(prob_data, len(group))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    fit_rho, _ = spearmanr(mean_fit_data, prob_data)
    rank_rho, _ = spearmanr(mean_rank_data, prob_data)

    fit_tau, _ = kendalltau(mean_fit_data, prob_data)
    rank_tau, _ = kendalltau(mean_rank_data, prob_data)

    fit_agree = np.mean(bin_fit_data == assign_data)

    corr_data.append({
        "id": id,
        "model": model_name,
        "n_annotators": len(group),
        "topic": topic,
        "topic_match_id": eval_data[id]["topic_match_id"],
        "fit_rho": fit_rho,
        "fit_tau": fit_tau,
        "rank_rho": rank_rho,
        "rank_tau": rank_tau,
        "fit_agree": fit_agree,
    })
corr_data = pd.DataFrame(corr_data)

#%% mann-whitney U tests
corr_data = corr_data.sort_values(["model", "topic_match_id"])
for model_a, model_b in combinations(["mallet", "ctm", "category-45"], 2):
    for metric in ["fit_rho", "rank_rho", "fit_tau", "rank_tau", "fit_agree"]:
        stat, pval = mannwhitneyu(corr_data[corr_data["model"] == model_a][metric].values, corr_data[corr_data["model"] == model_b][metric].values)
        alpha = 0.05 / 5 # Bonferroni correction for number of metrics
        sig = "*" if pval < alpha else ""
        print(f"{model_a} vs {model_b} | {metric} | {stat:.3f} | {pval:.3f}{sig}")


#%% then compute ndcg
ndcg_data = []
for model_name in model_fit_data.keys():
    fit_data = np.concatenate(model_fit_data[model_name], axis=0)
    rank_data = np.concatenate(model_rank_data[model_name], axis=0)
    prob_data = np.concatenate(model_prob_data[model_name], axis=0)
    
    fit_ndcg = ndcg_score(fit_data, prob_data)
    rank_ndcg = ndcg_score(rank_data, prob_data)

    print(f"{model_name} | Fit NDCG: {fit_ndcg:.5f} | Rank NDCG: {rank_ndcg:.5f}")
    ndcg_data.append({
        "model": model_name,
        "fit_ndcg": fit_ndcg,
        "rank_ndcg": rank_ndcg,
    })

ndcg_data = pd.DataFrame(ndcg_data)

#%%
corr_data["model"] = corr_data["model"].replace({"mallet": "Mallet", "ctm": "CTM", "category-45": "Labeled"})
# change data from wide to long, where each row is (model, fit/rank, rho/tau, value)
corr_plot_data = pd.melt(
    corr_data,
    id_vars=["model", "id", "topic_match_id"],
    value_vars=["fit_rho", "rank_rho", "fit_tau", "rank_tau", "fit_agree"],
    var_name="metric",
    value_name="Value",
)
corr_plot_data["score_type"] = corr_plot_data["metric"].str.split("_").str[0].str.capitalize()
corr_plot_data["coefficient"] = corr_plot_data["metric"].str.split("_").str[1].str.capitalize()
corr_plot_data["coefficient"] = corr_plot_data["coefficient"].astype("category").cat.reorder_categories(["Rho", "Tau", "Agree"])
corr_plot_data["model"] = corr_plot_data["model"].astype("category").cat.reorder_categories(["Mallet", "CTM", "Labeled"])
corr_plot_data["topic_match_id"] = corr_plot_data["topic_match_id"].astype("category") # can plot as color, but no real relationship model-to-model
corr_plot = (
    ggplot(corr_plot_data, aes(x="model", y="Value", color="model"))
    + geom_point()
    + scale_color_brewer(type="qual", palette=2)
    # score type on rows of facet, coefficient on columns
    + facet_grid("score_type ~ coefficient")
    + theme_classic()
    + theme(
        # remove x axis title
        axis_title_x=element_blank(),
        # remove y axis title
        axis_title_y=element_blank(),
        # remove legend
        legend_position="none",
        # make font a bit smaller
        text=element_text(size=7),
    )
)
corr_plot
# remove margins
#corr_plot.save("../figures/correlation_plot.pdf", dpi=300, width=4.65, height=3.5, bbox_inches='tight')

# %% NDCG plots
ndcg_data["model"] = ndcg_data["model"].replace({"mallet": "Mallet", "ctm": "CTM", "category-45": "Labeled"})
ndcg_plot_data = pd.melt(ndcg_data, id_vars=["model"], value_vars=["fit_ndcg", "rank_ndcg"], var_name="metric", value_name="value")
ndcg_plot_data["score_type"] = ndcg_plot_data["metric"].str.split("_").str[0].str.capitalize()
ndcg_plot_data["coefficient"] = "NDGC"
ndcg_plot_data["model"] = ndcg_plot_data["model"].astype("category").cat.reorder_categories(["Mallet", "CTM", "Labeled"])

# bar plot
ndcg_plot = (
    ggplot(ndcg_plot_data, aes(x="model", y="value", fill="model"))
    + geom_bar(stat="identity", position="dodge")
    + facet_grid("score_type ~ coefficient")
    + coord_cartesian(ylim=(0.89, 1))
    + scale_fill_brewer(type="qual", palette=2)
    + theme_classic()
    + theme(
        # remove x axis title
        axis_title_x=element_blank(),
        # remove y axis title
        axis_title_y=element_blank(),
        # remove legend
        legend_position="none",
        # make font a bit smaller
        text=element_text(size=7),
    )
)
ndcg_plot.save("../figures/ndcg_plot.pdf", dpi=300, width=1.55, height=3.5, bbox_inches='tight')
ndcg_plot

#%% get coherence 
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

with open("../data/document-data/wikitext/processed/labeled/vocab_15k/train.metadata.jsonl") as infile:
    docs = [json.loads(line)["tokenized_text"].split() for line in infile]
with open("../data/document-data/wikitext/processed/labeled/vocab_15k/test.metadata.jsonl") as infile:
    docs.extend([json.loads(line)["tokenized_text"].split() for line in infile])

vocab = sorted(set([word for doc in docs for word in doc]))
data_dict = Dictionary([vocab])

#%%
from tqdm import tqdm
for cluster_id, cluster_data in tqdm(eval_data.items()):
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
    eval_data[cluster_id]["npmi"] = mean
npmi_data = pd.DataFrame({'id': id, 'NPMI': data['npmi']} for id, data in eval_data.items())
corr_plot_data = corr_plot_data.merge(npmi_data, on="id")

#%% plot NPMI vs. correlation
corr_plot_data["color"] = "a"
npmi_corr_plot = (
    ggplot(corr_plot_data, aes(x="Value", y="NPMI", color="color"))
    + geom_point()
    + scale_color_brewer(type="qual", palette="Set1")
    + facet_grid("score_type ~ coefficient", scales="free")
    + theme_classic()
    + theme(
        # remove x axis title
        axis_title_x=element_blank(),
        # remove legend
        legend_position="none",
        # make font a bit smaller
        text=element_text(size=7),
    )
)
npmi_corr_plot.save("../figures/npmi_corr_plot.pdf", dpi=300, width=4.65, height=3.5, bbox_inches='tight')
npmi_corr_plot
# %%
corr_data = corr_data.merge(npmi_data, on="id")
pearsonr(corr_data.npmi, corr_data.rank_rho)
pearsonr(corr_data.npmi, corr_data.rank_rtau)

#%% ################
# REVISIONS
##################

corr_data = []
model_fit_data = {"mallet": [], "ctm": [], "category-45": []}
model_rank_data = {"mallet": [], "ctm": [], "category-45": []}
model_prob_data = {"mallet": [], "ctm": [], "category-45": []}

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

    for i in range(len(group)):
        # compute the correlations
        fit_rho, _ = spearmanr(fit_data[i], prob_data)
        rank_rho, _ = spearmanr(rank_data[i], prob_data)

        fit_tau, _ = kendalltau(fit_data[i], prob_data)
        rank_tau, _ = kendalltau(rank_data[i], prob_data)

        fit_ndcg = ndcg_score([fit_data[i]], [prob_data])
        rank_ndcg = ndcg_score([rank_data[i]], [prob_data])

        fit_agree = np.mean(bin_fit_data[i] == assign_data)
        #rank_agree = np.mean(assign_data[rank_data[i].argsort()[::-1][:assign_data.sum()]]) # how many of the top k are in the assigned k

        # to mean of other annotators
        other_mean_fit_data = np.mean(np.delete(fit_data, i, axis=0), axis=0)
        other_mean_rank_data = np.mean(np.delete(rank_data, i, axis=0), axis=0)

        fit_ia_rho, _ = spearmanr(fit_data[i], other_mean_fit_data)
        rank_ia_rho, _ = spearmanr(rank_data[i], other_mean_rank_data)

        fit_ia_tau, _ = kendalltau(fit_data[i], other_mean_fit_data)
        rank_ia_tau, _ = kendalltau(rank_data[i], other_mean_rank_data)

        corr_data.append({
            "id": id,
            "model": model_name,
            "n_annotators": len(group),
            "topic": topic,
            "topic_match_id": eval_data[id]["topic_match_id"],
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
            "rank_ia-tau": rank_ia_tau
        })
    
corr_data = pd.DataFrame(corr_data)

#%% mann-whitney U tests
corr_data = corr_data.sort_values(["model", "topic_match_id"])
for model_a, model_b in combinations(["mallet", "ctm", "category-45"], 2):
    for metric in ["fit_tau", "rank_tau", "fit_NDCG", "rank_NDCG", "fit_agree", "fit_ia-tau", "rank_ia-tau"]:
        stat, pval = mannwhitneyu(corr_data[corr_data["model"] == model_a][metric].values, corr_data[corr_data["model"] == model_b][metric].values)
        alpha = 0.05 / 7 # Bonferroni correction for number of metrics
        sig = "*" if pval < alpha else ""
        print(f"{model_a} vs {model_b} | {metric} | {stat:.3f} | {pval:.3f}{sig}")

#%%
corr_data["model"] = corr_data["model"].replace({"mallet": "Mallet", "ctm": "CTM", "category-45": "Labeled"})
# change data from wide to long, where each row is (model, fit/rank, rho/tau, value)
corr_plot_data = pd.melt(
    corr_data.drop(columns=["topic", "n_annotators", "annotator"]),
    id_vars=["model", "id", "topic_match_id"],
    var_name="metric",
    value_name="Value",
)

corr_plot_data["coefficient"] = corr_plot_data["metric"].str.split("_").str[-1]
corr_plot_data["score_type"] = corr_plot_data["metric"].str.split("_").str[0].str.capitalize()
corr_plot_data["model"] = corr_plot_data["model"].astype("category").cat.reorder_categories(["Mallet", "CTM", "Labeled"])
corr_plot_data["topic_match_id"] = corr_plot_data["topic_match_id"].astype("category") # can plot as color, but no real relationship model-to-model
corr_plot_data = corr_plot_data.loc[~corr_plot_data["coefficient"].str.contains("rho", case=False)]
corr_plot_data["coefficient"] = corr_plot_data["coefficient"].replace({"ia-tau": "Inter-Annotator Tau", "tau": "Model-Annotator Tau", "ndcg": "NDCG", "agree": "Binary Agreement"})
corr_plot_data["coefficient"] = corr_plot_data["coefficient"].astype("category").cat.reorder_categories(["Inter-Annotator Tau", "Model-Annotator Tau", "NDCG", "Binary Agreement"])

#corr_plot_data = corr_plot_data.loc[corr_plot_data.coefficient.str.contains("Tau")]
corr_plot = (
    ggplot(corr_plot_data, aes(x="model", y="Value", fill="model"))
    + geom_boxplot()
    + scale_fill_manual(values=["#66c2a5", "#fc8d62", "#b3b3b3"])
    # score type on rows of facet, coefficient on columns
    + facet_grid("score_type ~ coefficient")
    + theme_classic()
    + theme(
        # remove x axis title
        axis_title_x=element_blank(),
        # remove y axis title
        axis_title_y=element_blank(),
        # remove legend
        legend_position="none",
        # make font a bit smaller
        text=element_text(size=7),
    )
)
corr_plot
# remove margins
corr_plot.save("../figures/correlation_boxplot.pdf", dpi=300, width=6.2, height=3.5, bbox_inches='tight')


#%%

from plotnine import geom_smooth, scale_color_manual

corr_plot_data = corr_plot_data.merge(npmi_data, on="id")
corr_plot_data["color"] = "b"
corr_plot_data["coefficient"] = corr_plot_data["coefficient"].astype("category").cat.reorder_categories(["Inter-Annotator Tau", "Model-Annotator Tau", "NDCG", "Binary Agreement"])

npmi_corr_plot = (
    ggplot(corr_plot_data, aes(x="NPMI", y="Value", color="color"))
    + geom_point()
    #+ stat_summary(fun.data= mean_cl_normal) 
    + geom_smooth(method='lm')
    + scale_color_manual("#7570b3")
    + facet_grid("score_type ~ coefficient", scales="free")
    + theme_classic()
    + theme(
        # remove x axis title
        axis_title_y=element_blank(),
        # remove legend
        legend_position="none",
        # make font a bit smaller
        text=element_text(size=7),
    )
)
npmi_corr_plot
npmi_corr_plot.save("../figures/npmi_correlation_with_line.pdf", dpi=300, width=6.2, height=3.5, bbox_inches='tight')
npmi_corr_plot

for metric, df in corr_plot_data.groupby("metric"):
    rho, pval = pearsonr(df.Value, df.NPMI)
    print(f"{metric:11} {rho:0.3f} {pval:0.3f}")



#%%
### This is for error bars:
mean_corr_data = (
    corr_data.groupby(["id", "model", "topic", "topic_match_id", "n_annotators"])
             .agg(["mean", sem])
             .drop(columns=["annotator"])
)
mean_corr_data.columns = ["|".join(col) for col in mean_corr_data.columns.values]
mean_corr_data = mean_corr_data.reset_index()

mean_corr_data["model"] = mean_corr_data["model"].replace({"mallet": "Mallet", "ctm": "CTM", "category-45": "Labeled"})
# change data from wide to long, where each row is (model, fit/rank, rho/tau, value)
corr_plot_data = pd.melt(
    mean_corr_data.drop(columns=["topic", "n_annotators"]),
    id_vars=["model", "id", "topic_match_id"],
    var_name="metric",
    value_name="Value",
)

corr_plot_data["stat"] = corr_plot_data["metric"].str.split("|").str[1]
corr_plot_data["metric"] = corr_plot_data["metric"].str.split("|").str[0]
corr_plot_data = corr_plot_data.pivot(index=["model", "id", "topic_match_id", "metric"], columns="stat", values="Value")
corr_plot_data = corr_plot_data.reset_index()

corr_plot_data["score_type"] = corr_plot_data["metric"].str.split("_").str[0].str.capitalize()
corr_plot_data["coefficient"] = corr_plot_data["metric"].str.split("_").str[-1]
corr_plot_data["model"] = corr_plot_data["model"].astype("category").cat.reorder_categories(["Mallet", "CTM", "Labeled"])
corr_plot_data["topic_match_id"] = corr_plot_data["topic_match_id"].astype("category") # can plot as color, but no real relationship model-to-model

corr_plot_data = corr_plot_data.loc[~corr_plot_data["coefficient"].str.contains("rho", case=False)]
corr_plot_data["coefficient"] = corr_plot_data["coefficient"].replace({"ia-tau": "Inter-Annotator Tau", "tau": "Model-Annotator Tau", "ndcg": "NDCG", "agree": "Binary Agreement"})
corr_plot_data["coefficient"] = corr_plot_data["coefficient"].astype("category").cat.reorder_categories(["Inter-Annotator Tau", "Model-Annotator Tau", "NDCG", "Binary Agreement"])

corr_plot = (
    ggplot(corr_plot_data, aes(x="model", y="mean", color="model"))
    + geom_point(position=position_dodge2(width=0.5))
    + geom_errorbar(aes(ymin="mean-sem", ymax="mean+sem"), position=position_dodge2(width=0.5))
    + scale_color_brewer(type="qual", palette=2)
    # score type on rows of facet, coefficient on columns
    + facet_grid("score_type ~ coefficient")
    + theme_classic()
    + theme(
        # remove x axis title
        axis_title_x=element_blank(),
        # remove y axis title
        axis_title_y=element_blank(),
        # remove legend
        legend_position="none",
        # make font a bit smaller
        text=element_text(size=7),
    )
)
corr_plot