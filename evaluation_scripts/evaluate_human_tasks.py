#%% Imports
from itertools import combinations, groupby
import json 
import sys

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from scipy.stats import kendalltau

from irrCAC.raw import CAC

sys.path.append("../src/llm_eval")
from proxann.utils import (
    process_responses,
    collect_fit_rank_data,
    compute_correlations_one,
    compute_correlations_two,
    compute_agreement_per_topic
)

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
    ylim,
)

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

#%% Compute agreement
agreement_data_by_topic, _ = compute_agreement_per_topic(responses)
# if "wiki" in id, then "wiki", else "bills"
agreement_data_by_topic["dataset"] = ["wiki" if "wiki" in id else "bills" for id in agreement_data_by_topic["id"]]

#%% Make table
# make dataset categorical, ordered as wiki, bills
agreement_data_by_topic["dataset"] = pd.Categorical(agreement_data_by_topic["dataset"], categories=["wiki", "bills"], ordered=True)
# make model categorical: mallet, ctm, bertopic
agreement_data_by_topic["model"] = pd.Categorical(agreement_data_by_topic["model"], categories=["mallet", "ctm", "bertopic"], ordered=True)
print(
    agreement_data_by_topic
        .replace({
            "mallet": "Mallet",
            "ctm": "CTM",
            "bertopic": "BERTopic",
            "wiki": "i\texttt{Wiki}",
            "bills": "\\texttt{Bills}",
        })
        .groupby(["dataset", "model"])[["fit_alpha", "rank_alpha"]]
        .agg(lambda x: f"{x.mean():0.2f} ({x.std(ddof=1):0.2f})")
        .rename(columns={"fit_alpha": r"$\alpha$ Fit", "rank_alpha": r"$\alpha$ Rank"})
        .style
        .to_latex(hrules=True)
        .replace("dataset & model &  &  \\\\\n", "")
)

# %% Plot the correlations between humans and topic models

corr_mode2 = compute_correlations_two(responses)
corr_mode2["dataset"] = corr_mode2["id"].apply(lambda x: "wiki" if "wiki" in x else "bills")
corr_mode2["dataset"] = pd.Categorical(corr_mode2["dataset"], categories=["wiki", "bills"], ordered=True)
corr_mode2["model"] = corr_mode2["model"].replace({"mallet": "Mallet", "ctm": "CTM", "bertopic": "BERTopic"})
corr_mode2["model"] = pd.Categorical(corr_mode2["model"], categories=["Mallet", "CTM", "BERTopic"], ordered=True)

#%% change data from wide to long, where each row is (model, fit/rank, rho/tau, value)
for ds in ["wiki", "bills"]:
    corr_plot_data = pd.melt(
        corr_mode2.loc[corr_mode2.dataset == ds].drop(columns=["topic", "n_annotators", "annotator", "dataset"]),
        id_vars=["model", "id", "topic_match_id"],
        var_name="metric",
        value_name="Value",
    )

    corr_plot_data["coefficient"] = corr_plot_data["metric"].str.split("_").str[-1]
    corr_plot_data["score_type"] = corr_plot_data["metric"].str.split("_").str[0].str.capitalize()

    corr_plot_data["topic_match_id"] = corr_plot_data["topic_match_id"].astype("category") # can plot as color, but no real relationship model-to-model
    corr_plot_data = corr_plot_data.loc[~corr_plot_data["coefficient"].str.contains("rho", case=False)]
    corr_plot_data["coefficient"] = corr_plot_data["coefficient"].replace({"ia-tau": "Inter-Annotator Tau", "tau": "TM-Annotator Tau", "ndcg": "NDCG", "agree": "Binary Agreement"})
    corr_plot_data["coef_score_type"] = corr_plot_data["coefficient"] + " | " + corr_plot_data["score_type"]
    corr_plot_data["coefficient"] = corr_plot_data["coefficient"].astype("category").cat.reorder_categories(["Inter-Annotator Tau", "TM-Annotator Tau", "NDCG", "Binary Agreement"])

    ordered_coef_scores = [f"{cat} | {stype}" for cat in ["Inter-Annotator Tau", "TM-Annotator Tau", "NDCG"] for stype in ["Fit", "Rank"]]
    ordered_coef_scores += ["Binary Agreement | Fit"]
    corr_plot_data["coef_score_type"] = corr_plot_data["coef_score_type"].astype("category").cat.reorder_categories(ordered_coef_scores)
    # plot
    corr_plot_faceted = (
        ggplot(corr_plot_data, aes(x="model", y="Value", fill="model"))
        + geom_boxplot()
        + scale_fill_manual(values=["#66c2a5", "#fc8d62", "#8da0cb"])
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
    # remove margins
    corr_plot_faceted.save(f"../figures/correlation_boxplot_{ds}.pdf", dpi=300, width=6.2, height=3.5, bbox_inches='tight')

    # Make single-column version for length reasons, 
    for task in ["fit", "rank"]:
        corr_plot_data_task = corr_plot_data.loc[corr_plot_data.coefficient.str.contains("Tau")]
        corr_plot_data_task = corr_plot_data_task.loc[corr_plot_data.score_type == task.capitalize()]
        corr_plot_single_row = (
            ggplot(corr_plot_data_task, aes(x="model", y="Value", fill="model"))
            + geom_boxplot()
            + scale_fill_manual(values=["#66c2a5", "#fc8d62", "#8da0cb"])
            # make ylim from -0.24 to 1
            + ylim(-0.6, 1.0)
            # score type on rows of facet, coefficient on columns
            + facet_wrap("~ coefficient", ncol=4)
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
        corr_plot_single_row
        corr_plot_single_row.save(f"../figures/correlation_boxplot_single_row_{ds}_{task}.pdf", dpi=300, width=3.1, height=1.8, bbox_inches='tight')

    # make the topic ids categorical
    corr_plot_data = corr_plot_data.loc[corr_plot_data.coefficient.str.contains("Tau")]
    corr_plot_data = corr_plot_data.sort_values(by=["model", "Value"])
    corr_plot_data = corr_plot_data.loc[corr_plot_data.coefficient.str.contains("Inter-Annotator")]
    # first for model, mean for Value
    sorted_ids = (
        corr_plot_data.groupby(["id"])[["model", "Value"]]
        .agg({"model": "first", "Value": "median"})
        .sort_values(["model", "Value"])
        .index
    )

    corr_plot_data["id_cat"] = pd.Categorical(corr_plot_data["id"], categories=sorted_ids, ordered=True)

    corr_plot_faceted_by_topic = (
        ggplot(corr_plot_data, aes(x="id_cat", y="Value", fill="model"))
        + geom_boxplot()
        + scale_fill_manual(values=["#66c2a5", "#fc8d62", "#8da0cb"])
        # score type on rows of facet, coefficient on columns
        + facet_grid("~ coef_score_type")
        + theme_classic()
        + theme(
            # remove x axis title
            axis_title_x=element_blank(),
            # remove y axis title
            axis_title_y=element_blank(),
            # no legend title
            legend_title=element_blank(),
            # make font a bit smaller
            text=element_text(size=7),
            # remove x-axis tick labels
            axis_text_x=element_blank(),
        )
    )
    corr_plot_faceted_by_topic.save(f"../figures/correlation_boxplot_by_topic_{ds}.pdf", dpi=300, width=6.2, height=1.8, bbox_inches='tight')