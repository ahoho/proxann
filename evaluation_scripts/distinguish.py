#%% Imports
from itertools import combinations, groupby
import json 

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from scipy.stats import kendalltau

from proxann.utils import process_responses, collect_fit_rank_data, compute_correlations_one, compute_correlations_two

def read_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)

#%% Load the evaluation data and human responses
data_jsons = [
    "../data/json_out_from_submission/config_pilot_wiki.json",
    "../data/json_out_from_submission/config_pilot_wiki_part2.json",
    "../data/json_out_from_submission/config_bills_part1.json",
    "../data/json_out_from_submission/config_bills_part2.json",
]
response_csvs = [
    "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv",
    "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv",
]
start_date = "2024-12-06 09:00:00"

responses = {}
for csv in response_csvs:
    for topic_id, topic_responses in process_responses(csv, data_jsons, start_date=start_date, path_save=None, removal_condition="loose").items():
        if topic_responses:
            responses[topic_id] = topic_responses

_, _, _, corr_data = collect_fit_rank_data(responses)
corr_data = sorted(corr_data, key=lambda x: x["id"])
corr_ids = [x["id"] for x in corr_data]

#%% Load the model output data
llm_data_paths = {
    "gpt4o": [
        "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_gpt-4o-mini-2024-07-18_20241213_102655",
        "../data/llm_out/bills/q1_then_q3_dspy,q1_then_q2_dspy_gpt-4o-mini-2024-07-18_20241213_141345",
    ],
    # "gpt4o-t0.7-0": [
    #     "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_temp0.7_seed678_gpt-4o-2024-08-06_20250130_151207",
    # ],
    # "gpt4o-t0.7-1": [
    #     "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_temp0.7_seed801_gpt-4o-2024-08-06_20250130_141958",
    # ],
    "llama3.1": [
        "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_llama3.1:8b-instruct-q8_0_20241213_102533",
        "../data/llm_out/bills/q1_then_q3_dspy,q1_then_q2_dspy_llama3.1:8b-instruct-q8_0_20241213_133218",
    ],
    "llama3.1-70b": [
        "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_llama3.3:70b_20250126_094116",
        "../data/llm_out/bills/q1_then_q3_dspy,q1_then_q2_dspy_llama3.3:70b_20250126_095221",
    ],
    "qwen-32b": [
        "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_qwen:32b_20250126_223602",
        "../data/llm_out/bills/q1_then_q3_dspy,q1_then_q2_dspy_qwen:32b_20250126_223609",
    ],
}
llm_fits, llm_ranks = {}, {}
for llm, paths in llm_data_paths.items():
    fits, ranks = [], []
    for path in paths:
        fits.extend(read_json(f"{path}/llm_results_q2.json"))
        ranks.extend(read_json(f"{path}/llm_results_q3.json"))
    for item in fits:
        item["annotators"] = [llm]
    for item in ranks:
        item["annotators"] = [llm]

    fits = sorted([x for x in fits if x["id"] in corr_ids], key=lambda x: x["id"])
    ranks = sorted([x for x in ranks if x["id"] in corr_ids], key=lambda x: x["id"])

    llm_fits[llm] = fits
    llm_ranks[llm] = ranks

# %% compute correlations with the LLMs
task = "rank"
metric = "tau"
agg = "mean"

for llm in llm_data_paths:
    corrs_mode1_llm = compute_correlations_one(corr_data, llm_ranks[llm], llm_fits[llm], aggregation_method=agg)

    # rank
    for ds in ["wiki", "bills"]:
        corrs_ds = corrs_mode1_llm.loc[corrs_mode1_llm["id"].str.contains(ds)]
        llm_user_metric = corrs_ds[f"{task}_{metric}_users_{llm}"]
        llm_tm_metric = corrs_ds[f"{task}_{metric}_tm_{llm}"]
        user_tm_metric = corrs_ds[f"{task}_{metric}"]
        tau_topic_rank = kendalltau(llm_tm_metric, user_tm_metric, nan_policy="omit").statistic

        print(f"{llm:15}, {ds:5}  mean llm-user {llm_user_metric.mean():0.3f}, tau topics {tau_topic_rank:0.3f}")

# %% combine the LLMs
combined_ranks = []
combined_fits = []

models = ["gpt4o", "gpt4o-t0.7-0", "gpt4o-t0.7-1"]

for i in range(len(llm_fits["gpt4o"])):
    id = llm_fits["gpt4o"][i]["id"]
    for j, model in enumerate(models):
        llm_fit = llm_fits[model][i]["fit_data"]
        llm_rank = llm_ranks[model][i]["rank_data"]
        if j == 0:
            combined_fit = np.array(llm_fit)
            combined_rank = np.array(llm_rank)
        else:
            combined_fit += np.array(llm_fit)
            combined_rank += np.array(llm_rank)
    combined_fit = combined_fit / (j + 1) # TODO: make majority vote
    combined_rank = combined_rank / (j + 1) # TODO: re-do bradley terry on full pairwise collection

    combined_fits.append({
        "id": id,
        "annotators": ["combined"],
        "fit_data": combined_fit.tolist(),
    })
    combined_ranks.append({
        "id": id,
        "annotators": ["combined"],
        "rank_data": combined_rank.tolist(),
    })

corrs_mode1_combined = compute_correlations_one(corr_data, combined_ranks, combined_fits, aggregation_method=agg)

for ds in ["wiki", "bills"]:
    corrs_ds = corrs_mode1_combined.loc[corrs_mode1_combined["id"].str.contains(ds)]
    combined_user_metric = corrs_ds[f"{task}_{metric}_users_combined"]
    combined_tm_metric = corrs_ds[f"{task}_{metric}_tm_combined"]
    user_tm_metric = corrs_ds[f"{task}_{metric}"]

    tau_topic_rank = kendalltau(combined_tm_metric, user_tm_metric, nan_policy="omit").statistic
    print(f"combined mean, {ds:5} llm-user {combined_user_metric.mean():0.3f}, tau topics {tau_topic_rank:0.3f}")

# %% compute LOO correlations
def loo_from_corr_data(corr_data, seed=42, annotators_to_retain=None, keep_one=False):
    np.random.seed(seed)
    loo_data = []
    corr_data = sorted(corr_data, key=lambda x: x["id"])
    for topic_id, all_topic_data in groupby(corr_data, key=lambda x: x["id"]):
        for i, topic_data in enumerate(all_topic_data):
            if i == 0:
                # drop the same annotator across evaluations within the topic
                n = len(topic_data["fit_data"])
                idxs = np.arange(n)
                if annotators_to_retain is not None:
                    idx_to_exclude_from_sampling = annotators_to_retain[topic_id]
                else:
                    idx_to_exclude_from_sampling = -1
                if not keep_one:
                    loo_idx = np.random.choice(idxs[idxs != idx_to_exclude_from_sampling], 1)
                    keep_idxs = idxs[idxs != loo_idx]
                else:
                    loo_idx = idx_to_exclude_from_sampling
                    keep_idxs = idxs[idxs == loo_idx]
            topic_data = topic_data.copy()
            topic_data["fit_data"] = topic_data["fit_data"][keep_idxs]
            topic_data["rank_data"] = topic_data["rank_data"][keep_idxs]
            topic_data["dropped"] = loo_idx
            loo_data.append(topic_data)
    return loo_data

resample_data = []
for ds in ["wiki", "bills"]:
    for _ in tqdm(range(100)):
        corr_data_ds = [x for x in corr_data if ds in x["id"]]
        loo_corr_data_a = loo_from_corr_data(corr_data_ds, seed=None)
        annotators_to_retain = {x["id"]: x["dropped"] for x in loo_corr_data_a}

        corrs_mode1_loo_a = compute_correlations_one(loo_corr_data_a).sort_values("id")

        loo_corr_data_b = loo_from_corr_data(corr_data_ds, seed=None, annotators_to_retain=annotators_to_retain, keep_one=True)
        corrs_mode1_loo_b = compute_correlations_one(loo_corr_data_b).sort_values("id")

        resample_data.append({
            "ds": ds,
            "tau_of_rank_tau": kendalltau(corrs_mode1_loo_a["rank_tau"].values, corrs_mode1_loo_b["rank_tau"].values).statistic,
            "tau_of_fit_tau": kendalltau(corrs_mode1_loo_a["fit_tau"].values, corrs_mode1_loo_b["fit_tau"].values).statistic,
        })

        # TODO: reviewer suggestion to combine one human with the model

resample_data = pd.DataFrame(resample_data)
resample_data.groupby("ds").agg(["mean", "std"])

# %% Load a sentence embedding model
model = SentenceTransformer("all-mpnet-base-v2", device="mps")

# %% Tests of statistical indistinguishability
n_permutations = 1000
min_n_annotators = 4
conf_level = 0.95
llm_as_human = False
permutation_test_results = {llm: [] for llm in llm_data_paths}
np.random.seed(42)
rs = np.random.RandomState(42)

for llm in llm_data_paths:
    for topic_responses, topic_llm_fits, topic_llm_ranks in zip(tqdm(corr_data), llm_fits[llm], llm_ranks[llm]):
        assert(topic_responses["id"] == topic_llm_fits["id"] == topic_llm_ranks["id"])
        # Q1: get out the labels
        human_labels = topic_responses["label_data"]
        llm_label = topic_llm_fits["labels"][0] # they are always the same, pick the first

        human_embs = model.encode(human_labels)
        llm_emb = model.encode(llm_label)[None, :]

        # get label with shortest length
        shortest_label = min(human_labels+[llm_label], key=len)

        if len(human_labels) < min_n_annotators:
            continue

        # Q2: get out the fits
        human_fits = topic_responses["fit_data"]
        topic_llm_fits = topic_llm_fits["fit_data"][0]

        # Q3: get out the ranks
        human_ranks = topic_responses["rank_data"]
        topic_llm_ranks = topic_llm_ranks["rank_data"][0]

        # First, get all pairwise human comparisons
        human_pairs = []
        for a, b in combinations(range(len(human_labels)), 2):
            # Q1: compare labels
            # TODO: use a sentence-transformer to compare the text
            human_label_sim = cosine_similarity(human_embs[a, None], human_embs[b, None]).item()

            # Q2: compare fits
            human_fit_agree = ((human_fits[a] >= 4) == (human_fits[b] >= 4)).mean()

            # Q3: compare ranks
            human_rank_tau = kendalltau(human_ranks[a], human_ranks[b]).statistic

            # add to the list
            human_pairs.append({
                "topic_id": topic_responses["id"],
                "ann_idx_a": a,
                "ann_idx_b": b,
                "label_sim": human_label_sim,
                "fit_agree": human_fit_agree,
                "rank_tau": human_rank_tau,
            })
        human_pairs = pd.DataFrame(human_pairs)

        # Then, get all LLM-to-human comparisons
        llm_pairs = []
        for a in range(len(human_labels)):
            # Q1: compare labels
            llm_label_sim = cosine_similarity(human_embs[a, None], llm_emb).item()

            # Q2: compare fits
            llm_fit_agree = ((human_fits[a] >= 4) == topic_llm_fits).mean()

            # Q3: compare ranks
            llm_rank_tau = kendalltau(human_ranks[a], topic_llm_ranks).statistic

            # add to the list
            llm_pairs.append({
                "topic_id": topic_responses["id"],
                "ann_idx_a": a,
                "ann_idx_b": "llm",
                "label_sim": llm_label_sim,
                "fit_agree": llm_fit_agree,
                "rank_tau": llm_rank_tau,
            })
        llm_pairs = pd.DataFrame(llm_pairs)
        if llm_as_human:
            rand_human_idx = rs.choice(len(human_labels), replace=False)
            is_rand_human = (
                (human_pairs["ann_idx_a"] == rand_human_idx)
                | (human_pairs["ann_idx_b"] == rand_human_idx)
            )
            llm_pairs = human_pairs[is_rand_human]
            human_pairs = human_pairs[~is_rand_human]

        # Then run a permutation test to see if the LLM is indistinguishable from the humans
        # positive: LLM is better than humans
        observed_label_sim_diff =  llm_pairs["label_sim"].mean() - human_pairs["label_sim"].mean()
        observed_fit_agree_diff = llm_pairs["fit_agree"].mean() - human_pairs["fit_agree"].mean()
        observed_rank_tau_diff = llm_pairs["rank_tau"].mean() - human_pairs["rank_tau"].mean()

        # Bootstrap confidence intervals
        bootstrap_diffs = []
        for i in range(n_permutations):
            # Resample both groups with replacement
            boot_human = human_pairs.sample(frac=1, replace=True, random_state=rs)
            boot_llm = llm_pairs.sample(frac=1, replace=True, random_state=rs)

            boot_label_sim_diff = boot_llm["label_sim"].mean() - boot_human["label_sim"].mean()
            boot_fit_agree_diff = boot_llm["fit_agree"].mean() - boot_human["fit_agree"].mean()
            boot_rank_tau_diff = boot_llm["rank_tau"].mean() - boot_human["rank_tau"].mean()
        
            bootstrap_diffs.append({
                "label_sim": boot_label_sim_diff,
                "fit_agree": boot_fit_agree_diff,
                "rank_tau": boot_rank_tau_diff,
            })

        # Calculate confidence intervals
        bootstrap_diffs = pd.DataFrame(bootstrap_diffs)
        alpha = (1 - conf_level) / 2
        alpha_int = (alpha * 100, (1 - alpha) * 100)
        ci_label_sim = np.percentile(bootstrap_diffs["label_sim"], alpha_int)
        ci_fit_agree = np.percentile(bootstrap_diffs["fit_agree"], alpha_int)   
        ci_rank_tau = np.percentile(bootstrap_diffs["rank_tau"], alpha_int)

        perm_diffs = []
        for _ in range(n_permutations):
            perm_data = pd.concat([human_pairs, llm_pairs]).sample(frac=1, random_state=rs)
            perm_data_human = perm_data.iloc[:len(human_pairs)]
            perm_data_llm = perm_data.iloc[len(human_pairs):]

            perm_label_sim_diff = perm_data_llm["label_sim"].mean() - perm_data_human["label_sim"].mean()
            perm_fit_agree_diff = perm_data_llm["fit_agree"].mean() - perm_data_human["fit_agree"].mean()
            perm_rank_tau_diff = perm_data_llm["rank_tau"].mean() - perm_data_human["rank_tau"].mean()

            perm_diffs.append({
                "label_sim": perm_label_sim_diff,
                "fit_agree": perm_fit_agree_diff,
                "rank_tau": perm_rank_tau_diff,
            })

        perm_diffs = pd.DataFrame(perm_diffs)

        permutation_test_results[llm].append({
            "topic_id": topic_responses["id"],
            "dataset": "wiki" if "wikitext" in topic_responses["id"] else "bills",
            "shortest_label": shortest_label,
            "n_annotators": len(human_labels),
            "observed_label_sim_diff": observed_label_sim_diff,
            "observed_fit_agree_diff": observed_fit_agree_diff,
            "observed_rank_tau_diff": observed_rank_tau_diff,
            "ci_label_sim_low": ci_label_sim[0],
            "ci_label_sim_high": ci_label_sim[1],
            "ci_fit_agree_low": ci_fit_agree[0],
            "ci_fit_agree_high": ci_fit_agree[1],
            "ci_rank_tau_low": ci_rank_tau[0],
            "ci_rank_tau_high": ci_rank_tau[1],
            "label_sim_pval": (perm_diffs["label_sim"] <= observed_label_sim_diff).mean(),
            "fit_agree_pval": (perm_diffs["fit_agree"] <= observed_fit_agree_diff).mean(),
            "rank_tau_pval": (perm_diffs["rank_tau"] <= observed_rank_tau_diff).mean(),
        })

    permutation_test_results[llm] = pd.DataFrame(permutation_test_results[llm])

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def to_plot_df(
    results: pd.DataFrame,
    metric: str,
    dataset: str,
):
    """
    Convert permutation test results to a DataFrame for plotting.
    
    Args:
        results: DataFrame with permutation test results
        metric: metric to plot ('label_sim', 'fit_agree', or 'rank_tau')
    
    Returns:
        DataFrame with columns 'group', 'diff', 'p_value', 'ci_lower', 'ci_upper'
    """
    # Filter by metric
    results = results.dropna(subset=[f"observed_{metric}_diff"])
    results = results[results["dataset"] == dataset]

    # Create DataFrame
    plot_df = pd.DataFrame({
        'group': results['shortest_label'],
        'diff': results[f"observed_{metric}_diff"],
        'p_value': results[f"{metric}_pval"],
        'ci_lower': results[f"ci_{metric}_low"],
        'ci_upper': results[f"ci_{metric}_high"],
    })
    plot_df["group"] = plot_df["group"].replace(" and ", " & ", regex=True)

    return plot_df


def create_forest_plot(
    results: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 10),
    color: str = '#2b8cbe',
    sig_color: str = '#e41a1c',
    metric_name: str = "",
) -> plt.Figure:
    """
    Create a publication-ready vertical forest plot for agreement score differences.
    
    Args:
        results: DataFrame with columns:
            - group: group identifier
            - diff: mean difference (human-human - human-model)
            - p_value: permutation test p-value
            - ci_lower: lower confidence bound
            - ci_upper: upper confidence bound
            - n_pairs: total number of pairs (optional, for point sizing)
        figsize: figure dimensions
        color: color for non-significant points
        sig_color: color for significant points
    
    Returns:
        matplotlib Figure object
    """
    # set overall font size
    plt.rcParams.update({'font.size': 7})

    # Sort by effect size for better visualization
    results = results.sort_values('diff')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points and lines
    x_pos = np.arange(len(results))
    
    # Plot confidence intervals
    for i, (_, row) in enumerate(results.iterrows()):
        ax.vlines(
            x=i, 
            ymin=row['ci_lower'], 
            ymax=row['ci_upper'],
            color=sig_color if row['p_value'] < 0.05 else color,
            alpha=0.5,
            linewidth=1.5
        )
    
    # Plot points
    scatter = ax.scatter(
        x_pos,
        results['diff'],
        c=[sig_color if p < 0.05 else color for p in results['p_value']],
        s=10,
        zorder=10
    )
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Customize appearance
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results['group'], rotation=45, ha='right')
    
    ax.set_ylabel(f'Mean {metric_name} Difference\n(Human-LLM — Human-Human)')
        
    # Adjust layout for rotated labels
    plt.tight_layout()
    
    return fig

# Create and save plot
dataset = "wiki"
metric = "rank_tau"

plot_data = to_plot_df(permutation_test_results["gpt4o"], metric, dataset)
fig = create_forest_plot(plot_data, figsize=(6.29, 2.75), metric_name="Rank Tau")

# Save the figure
fig.savefig(f"../figures/forest_plot_{metric}_{dataset}.pdf", dpi=300, bbox_inches='tight')

# %% small table of results
perm_results = pd.concat([data.assign(llm=llm) for llm, data in permutation_test_results.items()])

# make dataset categorical, ordered as wiki, bills
perm_results["dataset"] = pd.Categorical(perm_results["dataset"], categories=["wiki", "bills"], ordered=True)
# make model categorical: gpt4o, llama3.1
perm_results["llm"] = pd.Categorical(perm_results["llm"], categories=["gpt4o", "llama3.1", "llama3.1-70b", "qwen-32b"], ordered=True)

perm_summary = (
    perm_results
        .groupby(["dataset", "llm"], observed=False)[["label_sim_pval", "fit_agree_pval", "rank_tau_pval"]]
        .apply(lambda x: (x >= 0.05).mean())
)

# make a latex table
print(
    perm_summary
        .rename(columns={"label_sim_pval": "Label Sim.", "fit_agree_pval": "Fit Acc.", "rank_tau_pval": "Rank $\\tau$"})
        .rename(
            index={
                "gpt4o": "GPT-4o",
                "llama3.1": "Llama3.1:8B",
                "llama3.1-70b": "Llama3.1:70B",
                "qwen-32b": "Qwen:32B",
                "wiki": "\\texttt{Wiki}",
                "bills": "\\texttt{Bills}"
            })
        .style
        .format("{:.0%}")
        .to_latex(
            hrules=True,
            label="tab:llm_perm_tests",
        )
        .replace("%", "\%")
        .replace("dataset & llm &  &  &  \\\\\n", "")
)
# %% same table in markdown
print(
    perm_summary
        .rename(columns={"label_sim_pval": "Label Sim.", "fit_agree_pval": "Fit Acc.", "rank_tau_pval": "Rank $\\tau$"})
        .rename(
            index={
                "gpt4o": "GPT-4o",
                "llama3.1": "Llama3.1:8B",
                "llama3.1-70b": "Llama3.1:70B",
                "qwen-32b": "Qwen:32B",
            })
        .reset_index()
        .round(2)
        .to_markdown(index=False)
)
# %%
