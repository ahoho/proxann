import json
import random
import boto3
import textwrap
from collections import Counter
from flask import Flask, jsonify, request

application = app = Flask(__name__)
app.config.from_file("_config.json", load=json.load)

# initialize the S3 client
s3 = boto3.client('s3')

def load_data_from_file(fpath):
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data

def save_obj_to_s3(obj, s3_key):
    """Save an object to S3 as a JSON file"""
    obj_json = json.dumps(obj)
    s3.put_object(Bucket=app.config["BUCKET_NAME"], Key=s3_key, Body=obj_json)


def load_obj_from_s3(s3_key):
    """Load a json file from S3"""
    obj = s3.get_object(Bucket=app.config["BUCKET_NAME"], Key=s3_key)
    return json.loads(obj['Body'].read().decode('utf-8'))


def load_counts(s3_key):
    """Load the counts from an S3 file"""
    try:
        counts = Counter(load_obj_from_s3(s3_key))
    except s3.exceptions.NoSuchKey:
        counts = Counter()
    return counts


def increase_counts(ids, s3_key, counts=None):
    """Increase the count for ids by 1"""
    if counts is None:
        counts = load_counts(s3_key)
    if not isinstance(ids, list):
        ids = [ids]
    for id in ids:
        counts[id] += 1
    save_obj_to_s3(counts, s3_key)
    return counts


@app.route("/api/reset_counts", methods=["POST"])
def reset_counts():
    """Reset the counts to zero"""
    s3_key = request.args.get("s3_key", None)
    if s3_key is None:
        return jsonify({"message": "No key provided"})
    if not load_counts(s3_key):
        return jsonify({"message": "No counts for this key"})
    save_obj_to_s3({}, s3_key)
    return jsonify({"message": "Counts reset"})


@app.route("/api/get_counts", methods=["GET"])
def get_counts():
    """Get the counts"""
    s3_key = request.args.get("s3_key", app.config["COUNTS_FILE"])
    counts = load_counts(s3_key)
    return jsonify(counts)


def docs_to_html_list(docs, li_style="", max_chars=None):
    """Make an HTML list of documents with collapsible sections"""
    max_chars = max_chars or 1_000_000
    outer_html = "<ul>{}</ul>"
    inner_html_template = f"""<li {li_style}>{{}}</li>"""
    inner_html = "".join([inner_html_template.format(textwrap.shorten(d, max_chars)) for d in docs])
    return outer_html.format(inner_html)


@app.route("/api/cluster_rank", methods=["GET"])
def get_cluster_to_rank():
    """
    Sample a cluster for ranking

    Expected format for the cluster data:
    {
        <model_run_identifier>: {
            <cluster_id>: {
                "topic_words": List[str], # sorted list, maybe limit to 100
                "exemplar_docs": List[str], # probable documents for cluster (e.g,., 5? or more, and can limit in the code)
                "eval_docs": [
                    # stratified sample of documents to evaluate for topic membership, from most probable to least probable
                    {"doc_id": int, "text": str, "prob": float, "assigned_to_k": bool},
                    ...
                ]
            },
            ...
        }
    }
    """
    # url format: /api/cluster_rank?num_words=10&num_questions=10
    num_words = int(request.args.get('num_words', 15)) # words in topic
    id = request.args.get('id', None)
    filter = request.args.get('filter', None) # filter by some string in the id
    max_chars = int(request.args.get('max_chars', 1_000_000))
    data_file = request.args.get('data_file', app.config["DATA_FILE"])

    counts = load_counts(app.config["COUNTS_FILE"]) # this is stored on S3
    
    # get the data for this cluster
    data = load_data_from_file(data_file)
    if filter is not None:
        data = {k: v for k, v in data.items() if filter in k}
    # want to have even sampling, so get cluster with the fewest counts, breaking ties randomly
    if id is None:
        id = min(data.keys(), key=lambda x: counts[x] + random.random())

    increase_counts(id, app.config["COUNTS_FILE"], counts)

    cluster_data = data[id]
    prepared_data = {}
    prepared_data["id"] = id
    prepared_data[f"top_words"] = " ".join(cluster_data[f"topic_words"][:num_words])
    prepared_data[f"exemplar_html"] = docs_to_html_list(
        [doc["text"] for doc in cluster_data[f"exemplar_docs"]],
        li_style='style="margin: 15px;"',
        max_chars=max_chars,
    )
    prepared_data["distractor_doc"] = textwrap.shorten(cluster_data["distractor_doc"]["text"], max_chars)
    for q_idx, eval_doc_data in enumerate(cluster_data["eval_docs"]):
        prepared_data[f"eval_id_{q_idx}"] = eval_doc_data["doc_id"]
        prepared_data[f"eval_prob_{q_idx}"] = eval_doc_data["prob"]
        prepared_data[f"eval_doc_{q_idx}"] = textwrap.shorten(eval_doc_data["text"], max_chars)
        prepared_data[f"eval_assigned_{q_idx}"] = eval_doc_data["assigned_to_k"]

    return jsonify(prepared_data)

if __name__ == '__main__':
    app.run(debug=False)