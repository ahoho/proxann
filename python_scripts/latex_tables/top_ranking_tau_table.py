def generate(
    data,
    left_columns=["Dataset", "Metric"],
    human_metrics= ["Fit", "Rank"],
    metrics_first_level=["Q2", "Q3"],
    metrics_second_level=["$C_{NPMI}$", "$C_{V}$"],
    llm_models=["gpt-4-o", "llama3.1"],
    datasets=["Wiki", "Bills"],
    transpose=False
):  
    """
    For each dataset... 
    - Row one has:
        * fit_tau_vs_npmi 
        * fit_tau_vs_cv
        * fit_tau_vs_fit_tau_tm_llmXXX ... (as many as there are LLM models)
        * fit_tau_vs_rank_tau_tm_llmXX ... (as many as there are LLM models)
    
    - Row two has:
        * rank_tau_vs_npmi
        * rank_tau_vs_cv
        * rank_tau_vs_fit_tau_tm_llmXX ... (as many as there are LLM models)
        * rank_tau_vs_rank_tau_tm_llmXX ... (as many as there are LLM models)
    """
    
    dict_metrics_first = {
        "Q2": "Step 2",
        "Q3": "Step 3"
    }
    
    if transpose:
        
        nr_left_columns = 2
        headers_first_level = (
            [""] * nr_left_columns # for the first two columns
            + [f"\multicolumn{{2}}{{c}}{{\\textsc{{{dataset.capitalize()}}}}}" for dataset in datasets]
        )
        headers_second_level = (
            ["", ""]
            + [metric.capitalize() for _ in datasets for metric in human_metrics]
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

        first_row = True
        # coherence
        for metric, metric_name in zip(["npmi", "cv"], metrics_second_level):
            row_label = rf"{'Coherence\n metrics' if first_row else ''} & {metric_name}"
            first_row = False
            
            row_values = []
            for dataset in datasets:
                for human_metric in human_metrics:
                    key = f"{human_metric.lower()}_tau_vs_{metric}"
                    value = data[dataset].get(key, 0.0)
                    row_values.append(f"{value:.2f}")

            latex_table += f"{row_label} & {' & '.join(row_values)} \\\\\n"

        latex_table += r"\midrule" + "\n"

        # llm metrics
        for metric_first in metrics_first_level:
            first_row = True
            for llm_model in llm_models:
                row_label = (
                    rf"{dict_metrics_first[metric_first] if first_row else ''} & \textsc{{{llm_model.capitalize()}}}"
                )
                first_row = False
                row_values = []

                for dataset in datasets:
                    for human_metric in human_metrics:
                        key = f"{human_metric.lower()}_tau_vs_{metric_first.lower()}_tm_{llm_model}"
                        value = data[dataset].get(key, 0.0)
                        row_values.append(f"{value:.2f}")

                latex_table += f"{row_label} & {' & '.join(row_values)} \\\\\n"
                
            latex_table += r"\midrule" + "\n"

        latex_table += "\\bottomrule\n\\end{tabular}"
                
        
    else:
        
        headers_first_level = (
            ["" for _ in left_columns]
            + ["" for _ in metrics_second_level]
            + [f"\multicolumn{{{len(llm_models)}}}{{c}}{{{dict_metrics_first[metric]}}}" for metric in metrics_first_level]
        )

        column_format = "ll" + "c" * len(metrics_second_level) + "c" * len(llm_models) * len(metrics_first_level)

        latex_table = rf"""
        \begin{{tabular}}{{@{{}}{column_format}@{{}}}}
        \toprule
        {" & ".join(headers_first_level)} \\
        {''.join([f'\cmidrule(lr){{{len(left_columns)+len(metrics_second_level)+i*len(llm_models)+1}-{len(left_columns)+len(metrics_second_level)+(i+1)*len(llm_models)}}}' for i in range(len(metrics_first_level))])}
        {" & " * len(left_columns)} {" & ".join(metrics_second_level)} & {" & ".join([f"\\textsc{{{model}}}" for _ in metrics_first_level for model in llm_models])} \\
        \midrule
        """
        
        for dataset in datasets:
            first_row = True
            for metric in human_metrics:
                metric = metric.lower()
                if first_row:
                    # Add \multirow for the dataset name
                    latex_table += rf"\multirow{{2}}{{*}}{{{f"\\textsc{{{dataset.capitalize()}}}"}}} & {metric.capitalize()} & "
                    first_row = False
                else:
                    latex_table += f" & {metric.capitalize()} & "
                
                npmi = data[dataset].get(f"{metric}_tau_vs_npmi", 0.0)
                cv = data[dataset].get(f"{metric}_tau_vs_cv", 0.0)
                
                vs_llms = []
                for llm_model in llm_models:
                    for metric_first in metrics_first_level:
                        key = f"{metric}_tau_vs_{metric_first.lower()}_tm_{llm_model}"
                        value = data[dataset].get(key, 0.0)
                        vs_llms.append(value)
                
                latex_table += f"{npmi:.2f} & {cv:.2f} & {' & '.join([f'{value:.2f}' for value in vs_llms])}{r' \\'}\n"
            
            latex_table += r"\midrule" + "\n"

        latex_table += "\\bottomrule\n\\end{tabular}"
    return latex_table



# Generar la tabla transpuesta
data = {
    "Wiki": {
        "fit_tau_vs_npmi": 1.0,
        "fit_tau_vs_cv": 1.5,
        "fit_tau_vs_q2_tm_gpt-4-o": 0.6,
        "fit_tau_vs_q2_tm_llama3.1": 0.8,
        "fit_tau_vs_q3_tm_gpt-4-o": 0.7,
        "fit_tau_vs_q3_tm_llama3.1": 0.9,
        "rank_tau_vs_npmi": 0.4,
        "rank_tau_vs_cv": 0.3,
        "rank_tau_vs_q2_tm_gpt-4-o": 0.5,
        "rank_tau_vs_q2_tm_llama3.1": 0.7,
        "rank_tau_vs_q3_tm_gpt-4-o": 0.6,
        "rank_tau_vs_q3_tm_llama3.1": 0.8,
    },
    "Bills": {
        "fit_tau_vs_npmi": 1.0,
        "fit_tau_vs_cv": 1.5,
        "fit_tau_vs_q2_tm_gpt-4-o": 0.6,
        "fit_tau_vs_q2_tm_llama3.1": 0.8,
        "fit_tau_vs_q3_tm_gpt-4-o": 0.7,
        "fit_tau_vs_q3_tm_llama3.1": 0.9,
        "rank_tau_vs_npmi": 0.4,
        "rank_tau_vs_cv": 0.3,
        "rank_tau_vs_q2_tm_gpt-4-o": 0.5,
        "rank_tau_vs_q2_tm_llama3.1": 0.7,
        "rank_tau_vs_q3_tm_gpt-4-o": 0.6,
        "rank_tau_vs_q3_tm_llama3.1": 0.8,
    },
}

latex_output_transposed = generate(data, transpose=True)
print(latex_output_transposed)