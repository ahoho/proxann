from collections import defaultdict
import pandas as pd
from scipy.stats import kendalltau
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_text,
    theme_classic,
    theme,
    element_text,
    element_blank,
    scale_color_manual,
    labs,
    scale_y_reverse,
    scale_x_discrete
)

import sys
sys.path.append("..")

from src.llm_eval.utils import process_responses, collect_fit_rank_data, compute_correlations_one

def generate(
    data,
    human_metrics= ["Human Fit", "Human Rank"],
    llm_metrics=["Step 2", "Step 3"],
    cohr_metrics=["$C_{NPMI}$", "$C_{V}$"],
    llm_models=["gpt-4-o", "llama3.1"],
    datasets=["Wiki", "Bills"],
):  
    """   
    Each column is kendall tau between topic-level metrics. "fit" is human binary agreement with the topic model assignments; "rank" is kendall tau (or ndcg) with the topic model thetas; "Q2" is binary agreement between LLM and topic model assignments; "Q3" is kendall tau (or ndcg) between LLM ranks and topic model thetas.
    """

    nr_left_columns = 2
    headers_first_level = (
        [""] * nr_left_columns # for the first two columns
        + [f"\multicolumn{{2}}{{c}}{{\\textsc{{{dataset.capitalize()}}}}}" for dataset in datasets]
    )
    headers_second_level = (
        ["", ""]
        + [metric for _ in datasets for metric in human_metrics]
    )

    column_format = "ll" + "c" * len(datasets) * len(human_metrics)

    latex_table = rf"""
    \begin{{tabular}}{{@{{}}{column_format}@{{}}}}
    \toprule
    {" & ".join(headers_first_level)} \\
    {"".join([
        f'\cmidrule(lr){{{nr_left_columns + i*len(human_metrics) + 1}-{nr_left_columns + i*len(human_metrics) + len(human_metrics)}}}' 
        for i in range(len(datasets))
    ])}
    {" & ".join(headers_second_level)} \\
    \midrule
    """
    
    # Pivot to structure data as needed
    aux = data.to_latex(index=False, float_format="{:.2f}".format).split("\\midrule\n")[1].split("\n\\bottomrule")[0].strip().split("\n")
    rows = [el.strip("\\").strip(" ") for el in aux]
    
    first_row = True
    # coherence
    for i, metric_name in enumerate(cohr_metrics):
        
        print(f"Index: {i}, Metric: {metric_name}, Metric Name: {metric_name}")
        
        row_label = rf"{'Coherence\n metrics' if first_row else ''} & {metric_name}"
        first_row = False
        
        row_values = rows[i]

        latex_table += f"{row_label} & {row_values} \\\\\n"

    latex_table += r"\midrule" + "\n"

    # llm metrics
    for i, metric_first in enumerate(llm_metrics):
        first_row = True
        for llm_model in llm_models:
            row_label = (
                rf"{metric_first if first_row else ''} & \textsc{{{llm_model.capitalize()}}}"
            )
            first_row = False
            row_values = rows[(i + 2)]

            latex_table += f"{row_label} & {row_values} \\\\\n"
            
        latex_table += r"\midrule" + "\n"

    latex_table += "\\bottomrule\n\\end{tabular}"
    
    return latex_table

data_jsons = [
    "../data/json_out/config_pilot_wiki.json",
    "../data/json_out/config_pilot_wiki_part2.json",
    "../data/json_out/config_bills_part1.json"
]
response_csvs = [
    "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+12,+2024_13.07.csv",
    "../data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv",
]

llm_data_paths = {
    "wiki": {
        "gpt4o": "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_gpt-4o-mini-2024-07-18_20241213_135628",
        "llama3.1": "../data/llm_out/wiki/q1_then_q3_dspy,q1_then_q2_dspy_llama3.1:8b-instruct-q8_0_20241213_135530"
    },
    "bills": {
        "gpt4o": "../data/llm_out/bills/q1_then_q3_dspy,q1_then_q2_dspy_gpt-4o-mini-2024-07-18_20241213_141345",
        "llama3.1": "../data/llm_out/bills/q1_then_q3_dspy,q1_then_q2_dspy_llama3.1:8b-instruct-q8_0_20241213_133218"
    }
}

cohr_path = "../data/all_cohrs.csv"

start_date = "2024-12-06 09:00:00"

responses = {}
for csv in response_csvs:
    for topic_id, topic_responses in process_responses(csv, data_jsons, start_date=start_date, path_save=None, removal_condition="loose").items():
        if topic_responses:
            responses[topic_id] = topic_responses

_, _, _, corr_data = collect_fit_rank_data(responses)

# read coherence metrics
cohrs = pd.read_csv(cohr_path).to_dict(orient="records")

corr_ids = [x["id"] for x in corr_data]
corr_metric = "tau"
agg = "mean"

# Obtener las claves internas del primer nivel del diccionario
row_names = ["npmi", "cv"] + [f"{metric}_{llm}" 
                              for llm in list(llm_data_paths[list(llm_data_paths.keys())[0]].keys()) 
                              for metric in ["fit", "rank"]]

column_names = [f"{metric}_{dataset}" for dataset in ["wiki", "bills"] for metric in ["fit", "rank"]]

results_df = pd.DataFrame(index=row_names, columns=column_names)

# Fill the DataFrame with calculated values
for dataset in ["wiki", "bills"]:
    print(f"Processing {dataset} dataset")
    llm_fits, llm_ranks = {}, {}
    for llm, path in llm_data_paths[dataset].items():
        fits = pd.read_json(f"{path}/llm_results_q2.json").to_dict(orient="records")
        ranks = pd.read_json(f"{path}/llm_results_q3.json").to_dict(orient="records")
        for item in fits:
            item["annotators"] = [llm]
        for item in ranks:
            item["annotators"] = [llm]
        llm_fits[llm] = fits
        llm_ranks[llm] = ranks
    
    for llm in llm_data_paths[dataset]:  # Iterate through each LLM
        
        dtset_corr_ids = [x for x in corr_ids if dataset in x]
    
        filtered_ranks = [x for x in llm_ranks[llm] if x["id"] in dtset_corr_ids]
        filtered_fits = [x for x in llm_fits[llm] if x["id"] in dtset_corr_ids]
        filtered_cohrs = [x for x in cohrs if x["id"] in dtset_corr_ids] 
        filtered_corr_data = [x for x in corr_data if x["id"] in dtset_corr_ids]
        
        # ensure that the lists are sorted by id        
        filtered_ranks = sorted(filtered_ranks, key=lambda x: x["id"])
        filtered_fits = sorted(filtered_fits, key=lambda x: x["id"])
        filtered_cohrs = sorted(filtered_cohrs, key=lambda x: x["id"])
        filtered_corr_data = sorted(filtered_corr_data, key=lambda x: x["id"])
        
        ids_fits = [x["id"] for x in filtered_fits]
        ids_ranks = [x["id"] for x in filtered_ranks]
        ids_corr = [x["id"] for x in filtered_corr_data]
                
        # check equality of ids
        assert ids_fits == ids_ranks == ids_corr
                
        npmi_cohrs = [x["npmi"] for x in filtered_cohrs]
        cv_cohrs = [x["cv"] for x in filtered_cohrs]

        corrs_mode1_llm = compute_correlations_one(filtered_corr_data, filtered_ranks, filtered_fits, aggregation_method=agg)
        
        ############################
        # TODO: this needs to be removed once all cohr data is ready
        # filter corrs_mode1_llm to have only the id that are in the cohr data
        corrs_mode1_llm = corrs_mode1_llm[corrs_mode1_llm["id"].isin([x["id"] for x in filtered_cohrs])]
        ############################

        for user_metric in ["fit", "rank"]:
            user_tm_metric = corrs_mode1_llm[f"{user_metric}_{corr_metric}"]

            # Correlate user_tm_metric with npmi and cv
            npmi_user_metric = kendalltau(npmi_cohrs, user_tm_metric, nan_policy="omit").statistic
            cv_user_metric = kendalltau(cv_cohrs, user_tm_metric, nan_policy="omit").statistic

            # Store the npmi and cv correlations in the DataFrame
            results_df.loc["npmi", f"{user_metric}_{dataset}"] = npmi_user_metric
            results_df.loc["cv", f"{user_metric}_{dataset}"] = cv_user_metric

            for llm_metric in ["fit", "rank"]:
                llm_tm_metric = corrs_mode1_llm[f"{llm_metric}_{corr_metric}_tm_{llm}"]

                # Correlate user_tm_metric with llm_tm_metric
                tau_topic_rank = kendalltau(llm_tm_metric, user_tm_metric, nan_policy="omit").statistic

                # Store the topic ranking correlation in the DataFrame
                results_df.loc[f"{llm_metric}_{llm}", f"{user_metric}_{dataset}"] = tau_topic_rank

print(results_df)
                
latex_output_transposed = generate(results_df)
print(latex_output_transposed)


### to generate figure
column_names = ["human_fit", "human_rank"] + ["npmi", "cv"] + [f"{llm}_{metric}" 
                              for llm in list(llm_data_paths[list(llm_data_paths.keys())[0]].keys()) 
                              for metric in ["fit", "rank"]]
row_names = ["mallet", "ctm", "bertopic"]

# create a dict of dataframes
dataset_statistics = defaultdict(pd.DataFrame)

for dataset in ["wiki", "bills"]:
    
    results_df = pd.DataFrame(index=row_names, columns=column_names)
    
    print(f"Processing {dataset} dataset")
    llm_fits, llm_ranks = {}, {}
    for llm, path in llm_data_paths[dataset].items():
        fits = pd.read_json(f"{path}/llm_results_q2.json").to_dict(orient="records")
        ranks = pd.read_json(f"{path}/llm_results_q3.json").to_dict(orient="records")
        for item in fits:
            item["annotators"] = [llm]
        for item in ranks:
            item["annotators"] = [llm]
        llm_fits[llm] = fits
        llm_ranks[llm] = ranks
    
    for llm in llm_data_paths[dataset]:  # Iterate through each LLM
        
        dtset_corr_ids = [x for x in corr_ids if dataset in x]
    
        filtered_ranks = [x for x in llm_ranks[llm] if x["id"] in dtset_corr_ids]
        filtered_fits = [x for x in llm_fits[llm] if x["id"] in dtset_corr_ids]
        filtered_corr_data = [x for x in corr_data if x["id"] in dtset_corr_ids]
        
        # ensure that the lists are sorted by id        
        filtered_ranks = sorted(filtered_ranks, key=lambda x: x["id"])
        filtered_fits = sorted(filtered_fits, key=lambda x: x["id"])
        filtered_cohrs = sorted(filtered_cohrs, key=lambda x: x["id"])
        filtered_corr_data = sorted(filtered_corr_data, key=lambda x: x["id"])

        corrs_mode1_llm = compute_correlations_one(filtered_corr_data, filtered_ranks, filtered_fits, aggregation_method=agg)

        for user_metric in ["fit", "rank"]:
            
            user_tm_metric_name = f"{user_metric}_{corr_metric}"            
            user_tm_metric = corrs_mode1_llm[["id", user_tm_metric_name]]
            
            # filter by model type
            for model_name in row_names:
                results_df.loc[model_name, f"human_{user_metric}"] = user_tm_metric[user_tm_metric["id"].str.contains(model_name)][user_tm_metric_name].mean()
                            
        for llm_metric in ["fit", "rank"]:
            
            llm_tm_metric_name = f"{llm_metric}_{corr_metric}_tm_{llm}"
            llm_tm_metric = corrs_mode1_llm[["id", llm_tm_metric_name]]
            
            # filter by model type
            for model_name in row_names:
                results_df.loc[model_name, f"{llm}_{llm_metric}"] = llm_tm_metric[llm_tm_metric["id"].str.contains(model_name)][llm_tm_metric_name].mean()            
        
        
    for cohr in ["npmi", "cv"]:
        
        cohrs = pd.read_csv(cohr_path)
        for model_name in row_names:
            # filter cohrs by model type
            cohrs_model = cohrs[cohrs["id"].str.contains(model_name)]
            results_df.loc[model_name, cohr] = cohrs_model[cohr].mean()
    
    results_df.index.name = "model"
    results_df = results_df.reset_index()
    dataset_statistics[dataset] = results_df

print("Wiki dataset")
print(dataset_statistics["wiki"])
print("Bills dataset")
print(dataset_statistics["bills"])

# generate graph for each dataset
for dtset in dataset_statistics.keys():
    
    dtset_df = dataset_statistics[dtset]    
    
    ranked_df = dtset_df.set_index('model').apply(lambda col: col.rank(ascending=False, method='min')) # rank according to each metric

    data = ranked_df.reset_index().melt(id_vars='model', var_name='metric', value_name='score') # convert to longformat

    data['label'] = data['model'].where(data['metric'] == "human_fit") # additional column to put the label in human_fit
    data['label'] = data['label'].map({
        'mallet': 'MALLET',
        'ctm': 'CTM',
        'bertopic': 'BERTopic'
    })

    desired_order = ['human_fit', 'human_rank', 'npmi', 'cv', 'gpt4o_fit', 'gpt4o_rank', 'llama3.1_fit', 'llama3.1_rank'] # force the order of the metrics
    data['metric'] = pd.Categorical(data['metric'], categories=desired_order, ordered=True)

    metric_labels = {
        'human_fit': 'Human\nFit',
        'human_rank': 'Human\nRank',
        'npmi': '$C_{NPMI}$',
        'cv': '$C_V$',
        'gpt4o_fit': 'GPT-4\nFit',
        'gpt4o_rank': 'GPT-4\nRank',
        'llama3.1_fit': 'LLaMA 3.1\nFit',
        'llama3.1_rank': 'LLaMA 3.1\nRank'
    }
    colors = {
        'mallet': '#66c2a5',
        'ctm': '#fc8d62',   
        'bertopic': '#8da0cb'
    }

    plot = (
        ggplot(data, aes(x='metric', y='score', color='model', shape='model'))
        + geom_point(size=6, stroke=1.5)  # symbols size
        + geom_text(
            aes(label='label'),  # labels only "human_fit"
            na_rm=True,  # remove nan
            nudge_y=0.3,  # move label up (y)
            nudge_x=-0.1, # move label left (x)
            size=12  # size of the labels
        )
        + scale_color_manual(values=colors) 
        + scale_y_reverse(breaks=[1, 2, 3], labels=[1, 2, 3], limits=[3.3, 0.7])  # y = rankings
        + scale_x_discrete(labels=metric_labels)  # format x-axis labels
        + labs(y='Ranking (1 = Best)', x='Metric', color=None, shape=None) 
        + theme_classic() 
        + theme(
            figure_size=(8, 4),
            axis_text_x=element_text(size=14, rotation=45, hjust=1),
            axis_text_y=element_text(size=14),
            axis_title=element_text(size=14),
            legend_position='none', # remove legend
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            axis_title_x=element_blank(), # remove x-axis caption
        )
    )
    
    plot.save(f"../figures/all_metrics_models_comparisson_{dtset}.pdf", dpi=300, bbox_inches='tight')