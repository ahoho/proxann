import re
import json
import argparse
from pathlib import Path
from zipfile import ZipFile

from annotation_server import load_data_from_file

def clean_wikitext(text):
    # cut text after first section
    text = text.split(" = = ")[0]
    # remove the title (follows first " = "), but only if it's short (otherwise title could be missing)
    if 1 < text.find(" = ") < 50:
        text = text.split(" = ")[-1]

    # remove spaces to left of punctuation and close brackets
    text = re.sub(r'\s([\)\].,!?])', r"\1", text)
    # remove spaces to right of open brackets
    text = re.sub(r'([\[\(])\s', r"\1", text)
    # s ' -> s'
    text = text.replace("s '", "s'")
    # n 't -> n't
    text = text.replace("n 't", "n't")
    # ' ve -> 've
    text = text.replace("' ve", "'ve")
    # ' re -> 're
    text = text.replace("' re", "'re")
    # ' ll -> 'll
    text = text.replace("' ll", "'ll")
    # <word> 's -> <word>'s
    text = re.sub(r'(\w) \'s', r"\1's", text)
    # <words> \u2019 s -> <word>'s
    text = re.sub(r'(\w) \u2019 s', r"\1's", text)

    # I 'm -> I'm
    text = text.replace("I 'm", "I'm")
    # convert " <text> " to "<text>"
    text = re.sub(r'"\s([^"]+)\s"', r'"\1"', text)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_data_json_files", type=str, nargs="+")
    parser.add_argument("--output_fpath", default="flask_app.zip", type=str)
    args = parser.parse_args()

    # slightly reformat the data
    complete_data = {} 
    for fpath in args.annotation_data_json_files:
        raw_data = load_data_from_file(fpath)
        data = {
            f"{model_id}/{cluster_id}": cluster_data
            for model_id, model_data in raw_data.items()
            for cluster_id, cluster_data in model_data.items()
        }
        for id in data:
            cluster_data = data[id]
            cluster_data["distractor_doc"]["text"] = clean_wikitext(cluster_data["distractor_doc"]["text"])

            for doc_data in cluster_data["exemplar_docs"]:
                doc_data["text"] = clean_wikitext(doc_data["text"])
            for doc_data in cluster_data["eval_docs"]:
                doc_data["text"] = clean_wikitext(doc_data["text"])
        complete_data.update(data)

    # write the data to a file
    with ZipFile(args.output_fpath, "w") as zipfile:
        zipfile.writestr("_cluster_rank_data.json", json.dumps(complete_data, indent=2))
        # copy the requirements.txt and annotation_server.py in the same directory
        zipfile.write("requirements.txt")
        zipfile.write("annotation_server.py", "application.py")
        zipfile.write("_config.json")