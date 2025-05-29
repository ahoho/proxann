#%%
import numpy as np
from pathlib import Path
import json
import sys

from proxann.utils import process_responses, collect_fit_rank_data

def read_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)
    
data_jsons = [
    "../data/json_out/config_pilot_wiki.json",
    "../data/json_out/config_pilot_wiki_part2.json",
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
corr_data = {item["id"]: item for item in corr_data}

#%% Load the model output data
base_path = Path("../data/camera_ready_llm_out/mean/")
model_output_paths = [
    sorted(Path(base_path, "wiki/gpt-4o-2024-08-06/").glob("*"))[0],
    sorted(Path(base_path, "bills/gpt-4o-2024-08-06/").glob("*"))[0],
]
llm_fit_data = {}
for path in model_output_paths:
    llm_out = read_json(f"{path}/llm_results_q2.json")
    for item in llm_out:
        llm_fit_data[item["id"]] = item

# %%
disagreeable_docs = []
agree_th = 2
disagree_th = 2

for k in corr_data:
    corr_topic = corr_data[k]
    llm_fit_topic = llm_fit_data[k]
    responses_topic = responses[k][0]
    n_docs = len(responses_topic["eval_docs"])
    for i in range(n_docs):
        llm_fit_doc = llm_fit_topic["fit_data"][0][i]
        human_fit_docs = corr_topic["fit_data"][:, i]
        eval_doc = responses_topic["eval_docs"][i]

        human_diff = np.abs(human_fit_docs.min() - human_fit_docs.max())
        llm_diff = np.abs(llm_fit_doc - human_fit_docs.mean())
        if human_diff <= agree_th and llm_diff > disagree_th:
            print(f"human diff: {human_diff:.2f}, llm diff: {llm_diff:.2f}")
            disagreeable_docs.append({
                "topic_id": k,
                "doc_id": eval_doc["doc_id"],
                "human_fit": human_fit_docs,
                "llm_fit": llm_fit_doc,
                "human_labels": corr_topic["label_data"],
                "llm_label": llm_fit_topic["labels"][0],
                "text": eval_doc["text"],
            })

print(f"Found {len(disagreeable_docs)} disagreeable documents")