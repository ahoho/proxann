#%% Imports
import json 
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy.stats import kendalltau, pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

from proxann.utils import process_responses, collect_fit_rank_data, compute_correlations, compute_correlations_two

def read_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)

#%% Load the evaluation data and human responses
data_jsons = [
    "../data/json_out/config_wiki_part1.json",
    "../data/json_out/config_wiki_part2.json",
    "../data/json_out/config_bills_part1.json",
    "../data/json_out/config_bills_part2.json",
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
base_path = Path("../data/llm_out/mean/")
llm_data_patterns = {
    "gpt-4o": {
        "wiki": list(Path(base_path, "wiki/gpt-4o-2024-08-06/").glob("*")),
        "bills": list(Path(base_path, "bills/gpt-4o-2024-08-06/").glob("*")),
    },
    "llama-8b": {
        "wiki": list(Path(base_path, "wiki/Meta-Llama-3.1-8B-Instruct").glob("*")),
        "bills": list(Path(base_path, "bills/Meta-Llama-3.1-8B-Instruct").glob("*")),
    },
    "llama-70b": {
        "wiki": list(Path(base_path, "wiki/llama-3.3-70b-instruct-awq/").glob("*")),
        "bills": list(Path(base_path, "bills/llama-3.3-70b-instruct-awq/").glob("*")),
    },
    "qwen-2.5-72b": {
        "wiki": list(Path(base_path, "wiki/Qwen2.5-72B-Instruct-AWQ/").glob("*")),
        "bills": list(Path(base_path, "bills/Qwen2.5-72B-Instruct-AWQ/").glob("*")),
    },
    "qwen-3-8b": {
        "wiki": list(Path(base_path, "wiki/Qwen3-8B/").glob("*")),
        "bills": list(Path(base_path, "bills/Qwen3-8B/").glob("*")),
    },
    "qwen-3-32b": {
        "wiki": list(Path(base_path, "wiki/Qwen3-32B/").glob("*")),
        "bills": list(Path(base_path, "bills/Qwen3-32B/").glob("*")),
    },
}
llm_fits, llm_ranks = {}, {}

#%%
for llm, paths_by_ds in llm_data_patterns.items():
    llm_fits[llm] = defaultdict(list)
    llm_ranks[llm] = defaultdict(list)

    for dataset, paths in paths_by_ds.items():
        fits_, ranks_ = [], []
        # iterate over all seeds
        for seed, path in enumerate(paths):
            fits_seed = read_json(f"{path}/llm_results_q2.json")
            ranks_seed = read_json(f"{path}/llm_results_q3.json")

            # point is to move from "seed by topic"
            # [[topic_0_seed_0, topic_1_seed_0, ...], [topic_0_seed_1, topic_1_seed_1, ...]]
            # to "topic by seed"
            # [[topic_0_seed_0, topic_0_seed_1, ...], [topic_1_seed_0, topic_1_seed_1, ...]]
            for i, (fit_item, rank_item) in enumerate(zip(fits_seed, ranks_seed)):
                assert(fit_item["id"] == rank_item["id"])
                if seed == 0:
                    fits_.append([fit_item])
                    ranks_.append([rank_item])
                else:
                    fits_[i].append(fit_item)
                    ranks_[i].append(rank_item)

        # then we can average over all seeds
        for fit_item, rank_item in zip(fits_, ranks_):
            id = fit_item[0]["id"]
            llm_fits[llm][dataset].append({
                "id": id,
                "annotators": [llm],
                "fit_data": [np.mean([x["fit_data"][0] for x in fit_item], axis=0).tolist()],
            })
            llm_ranks[llm][dataset].append({
                "id": id,
                "annotators": [llm],
                "rank_data": [np.mean([x["rank_data"][0] for x in rank_item], axis=0).tolist()],
            })

#%% Individual evaluation
task = "fit"
metric = "tau"
agg = "mean"

for model in llm_data_patterns:
    for ds in ["wiki", "bills"]:
        corrs_ds = compute_correlations(
            corr_data=corr_data,
            rank_llm_data=llm_ranks[model][ds],
            fit_llm_data=llm_fits[model][ds],
            aggregation_method=agg,
            fit_threshold_user=4, # 3 may be better?
            fit_threshold_llm=4,
            rescale_ndcg=True,
            binarize_tm_probs=True,
        )
        combined_user_metric = corrs_ds[f"{task}_{metric}_users_{model}"]
        combined_tm_metric = corrs_ds[f"{task}_{metric}_tm_{model}"]
        user_tm_metric = corrs_ds[f"{task}_{metric}"]

        tau_topic_rank = kendalltau(combined_tm_metric.values, user_tm_metric.values, nan_policy="omit").statistic
        print(f"{model:15}, {ds:5} llm-user {combined_user_metric.mean():0.3f}, tau topics {tau_topic_rank:0.3f}")
    
#%% loop and create a summary dataframe
summary = []
n_resample_annotators = 1
n_resample_topics = 50
metric = "tau"
models_to_retain = ["llama-8b", "qwen-3-8b", "qwen-3-32b", "llama-70b", "qwen-2.5-72b", "gpt-4o"]

total = len(models_to_retain) * 2 * n_resample_annotators * n_resample_topics
with tqdm(total=total, desc="Computing correlations") as pbar:
    for model in models_to_retain:
        rng = np.random.RandomState(42)
        for ds in ["wiki", "bills"]:
            for i in range(n_resample_annotators):
                corrs_ds = compute_correlations(
                    corr_data=corr_data,
                    rank_llm_data=llm_ranks[model][ds],
                    fit_llm_data=llm_fits[model][ds],
                    aggregation_method=agg,
                    fit_threshold_user=4, # 3 may be better?
                    fit_threshold_llm=4,
                    rescale_ndcg=True,
                    bootstrap_annotators=n_resample_annotators > 1,
                    seed=rng._bit_generator,
                )
                for j in range(n_resample_topics):
                    if n_resample_topics > 1:
                        corrs_ds_j = corrs_ds.sample(frac=1, random_state=rng._bit_generator, replace=True)
                    else:
                        corrs_ds_j = corrs_ds
                    for task in ["fit", "rank"]:
                        combined_user_metric = corrs_ds_j[f"{task}_{metric}_users_{model}"]
                        combined_tm_metric = corrs_ds_j[f"{task}_{metric}_tm_{model}"]
                        user_tm_metric = corrs_ds_j[f"{task}_{metric}"]

                        tau_topic_rank = kendalltau(combined_tm_metric.values, user_tm_metric.values, nan_policy="omit").statistic
                        summary.append({
                            "model": model,
                            "dataset": ds,
                            "task": task,
                            "tau_topic_rank": tau_topic_rank,
                            f"user_llm_{metric}": combined_user_metric.mean(),
                            "ann_i": i,
                            "topic_i": j,
                        })
                    pbar.update(1)

summary = pd.DataFrame(summary)


#%% retrieve human-human correlations as a comparison
human_to_human_corr = compute_correlations_two(responses)
human_to_human_corr["dataset"] = human_to_human_corr["id"].apply(lambda x: "wiki" if "wiki" in x else "bills")
human_df = human_to_human_corr.groupby("dataset")[["fit_ia-tau", "rank_ia-tau"]].mean()


#%% make into a bar chart

# Set style for more beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
# set font to Helvetica Neue
plt.rcParams['font.family'] = 'Helvetica'

# Assuming df is your original dataframe
df = summary

# Define model order by size and nice labels
model_order = ["llama-8b", "qwen-3-8b", "qwen-3-32b", "llama-70b", "qwen-2.5-72b", "gpt-4o"]
model_labels = ['Llama-3.1-8B', 'Qwen-3-8B', 'Qwen-3-32B', 'Llama-3.3-70B', 'Qwen-2.5-72B', 'GPT-4o']
model_mapping = dict(zip(model_order, model_labels))

# Calculate aggregated statistics with bootstrapped CIs
grouped = df.groupby(['model', 'dataset', 'task'])['user_llm_tau'].agg([
    'mean',
    lambda x: np.percentile(x, 2.5),  # Lower CI
    lambda x: np.percentile(x, 97.5)   # Upper CI
]).round(3)
grouped.columns = ['mean', 'ci_lower', 'ci_upper']
grouped = grouped.reset_index()

# Calculate error bars
grouped['err_lower'] = grouped['mean'] - grouped['ci_lower']
grouped['err_upper'] = grouped['ci_upper'] - grouped['mean']

# Create figure with two columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Use ColorBrewer colors - Set2 palette for models
colors = ['#a6bddb', '#c2e699',  '#78c679', '#2b8cbe', '#238443', '#c994c7']  # ColorBrewer Set2
color_mapping = dict(zip(model_order, colors))

# Task order (Fit before Rank)
tasks = ['fit', 'rank']
task_labels = ['Fit Step', 'Rank Step']
datasets = ['wiki', 'bills']
dataset_labels = {'wiki': 'Wiki', 'bills': 'Bills'}

# Width of bars and positions
bar_width = 0.15
n_models = len(model_order)
n_tasks = len(tasks)

for idx, (ax, dataset) in enumerate([(ax1, 'wiki'), (ax2, 'bills')]):
    # Filter data for this dataset
    data_subset = grouped[grouped['dataset'] == dataset]
    
    # X positions for tasks
    x_base = np.arange(n_tasks)
    
    # Plot bars for each model
    for i, model in enumerate(model_order):
        model_data = data_subset[data_subset['model'] == model]
        
        means = []
        err_lower = []
        err_upper = []
        
        for task in tasks:
            task_data = model_data[model_data['task'] == task]
            if not task_data.empty:
                means.append(task_data['mean'].values[0])
                err_lower.append(task_data['err_lower'].values[0])
                err_upper.append(task_data['err_upper'].values[0])
            else:
                means.append(0)
                err_lower.append(0)
                err_upper.append(0)
        
        # Calculate offset for grouped bars
        offset = (i - n_models/2 + 0.5) * bar_width
        
        # Plot bars with error bars
        bars = ax.bar(x_base + offset, means, bar_width, 
                      label=model_mapping[model] if idx == 0 else "",
                      color=color_mapping[model],
                      yerr=[err_lower, err_upper],
                      capsize=2,
                      # make error bars lighter
                      error_kw={'linewidth': 0.5, 'capthick': 0.5, 'ecolor': "#686868"},
                      edgecolor='white',
                      linewidth=0.5)
    
    # Customize subplot
    if idx == 0:
        ax.set_ylabel("Human-LLM Correlations (Tau)", fontsize=7.5, fontweight='medium')
    if idx == 1:
        ax.set_yticklabels([])
    
    ax.text(0.5, 0.99, dataset_labels[dataset], transform=ax.transAxes, 
        ha='center', va='top', fontsize=7.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels(task_labels, fontsize=7.5)
    ax.set_ylim(0, 0.75)
    
    # Remove all spines except bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Customize y-axis
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='y', labelsize=6, colors='#666666')
    ax.tick_params(axis='x', labelsize=6)
    
    # Remove horizontal grid lines
    ax.grid(False)
    
    # Add subtle vertical lines at y-ticks
    for y in ax.get_yticks():
        if y > 0:
            ax.axhline(y=y, color='#EEEEEE', linewidth=0.8, zorder=0)
    
    # Make bottom spine thinner and lighter
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#CCCCCC')

        # Add human-to-human correlation lines
    if dataset in human_df.index:
        # Add dashed line for fit task
        fit_value = human_df.loc[dataset, 'fit_ia-tau']
        ax.axhline(y=fit_value, color='#333333', linestyle='--', linewidth=0.75, 
                   alpha=0.7, zorder=10, xmin=0.05, xmax=0.5)
        
        # Add dashed line for rank task
        rank_value = human_df.loc[dataset, 'rank_ia-tau']
        ax.axhline(y=rank_value, color='#666666', linestyle='--', linewidth=0.75, 
                   alpha=0.7, zorder=10, xmin=0.525, xmax=.975)
        
# Add legend to the right of the second subplot
handles, labels = ax1.get_legend_handles_labels()
legend = ax2.legend(handles, labels, 
                   loc='center left', 
                   bbox_to_anchor=(.95, 0.5),
                   frameon=False,
                   fancybox=False,
                   shadow=False,
                   fontsize=7.5,
                   title_fontsize=7.5)

# Style the legend
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#CCCCCC')
legend.get_frame().set_linewidth(0.8)
legend.get_title().set_fontweight('bold')

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Set background color
fig.patch.set_facecolor('white')
for ax in [ax1, ax2]:
    ax.set_facecolor('white')

# Save figure
fig.set_size_inches(6.3, 1.6)
plt.savefig('../figures/human_llm_comparison_barplot.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

#%% big correlation plot (not in paper)
model = "gpt-4o"
ds = "bills"
task = "rank"
plot_df = []
if task == "rank":
    llm_data = llm_ranks
else:
    llm_data = llm_fits

for item in llm_data[model][ds]:
    human_item = next(x for x in corr_data if x["id"] == item["id"])
    human_score_i = human_item[f"{task}_data"].mean(0)
    llm_scores = item[f"{task}_data"][0]
    #for human_score_i in human_scores:
    for human_doc_score, llm_doc_score in zip(human_score_i, llm_scores):
        plot_df.append({
            "id": item["id"],
            "score_human": human_doc_score,
            "score_llm": llm_doc_score,
        })

plot_df = pd.DataFrame(plot_df)
plot_df.to_csv(f"../figures/{model}_{ds}_{task}_scores.csv", index=False)


plt.scatter(plot_df["score_llm"], plot_df["score_human"],  alpha=0.5)
plt.ylabel(f"Human {task.capitalize()}")
plt.xlabel(f"LLM {task.capitalize()}")
plt.title(f"{model} {ds} {task}")
plt.show()
print(kendalltau(plot_df["score_llm"], plot_df["score_human"], nan_policy="omit").statistic)
print(pearsonr(plot_df["score_llm"], plot_df["score_human"]))