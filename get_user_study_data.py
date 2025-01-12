"""Extracts information from a set of topic models provided in a configuration file and saves the output in a JSON file.

From each model, as many topics as specified in the 'n_matches' parameter of the configuration file are kept. The topics are selected using the iterative matching algorithm implemented in the TopicSelector class. 

For each of the selected topics, the top words, exemplar documents, evaluation documents, and a distractor document are extracted.

Then, using the path of the model as key, the output of all models is grouped in a dictionary and save it in a JSON file.
"""

import argparse
import configparser
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from src.user_study_data_collector.jsonfy.topic_json_formatter import TopicJsonFormatter
from src.user_study_data_collector.topics_docs_selection.topic_selector import TopicSelector
from src.utils.utils import load_vocab_from_txt


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config", help="Path to the config file", required=False,
        default="config/config_pilot.conf")

    args = argparser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    ############################
    # Topic selection          #
    ############################
    N = int(config['all']['n_matches'])
    top_words_display = int(config['all']['top_words_display'])

    remove_topic_ids = []
    all_topic_keys = []
    for model in config.sections():
        # Skip all section (configuration for all models)
        if model == 'all':
            continue
        model_config = config[model]
        # if trained with this repo code, we only need the model path
        if model_config.getboolean('trained_with_thetas_eval'):
            model_path = model_config['model_path']
            # Load vocab dictionaries
            vocab_w2id = {}
            with (pathlib.Path(model_path)/'vocab.txt').open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    wd = line.strip()
                    vocab_w2id[wd] = i
            betas = np.load(pathlib.Path(model_path) / "betas.npy")
        else:
            vocab_path = model_config['vocab_path']
            if vocab_path.endswith(".json"):
                with open(vocab_path) as infile:
                    vocab_w2id = json.load(infile)

            elif vocab_path.endswith()(".txt"):
                vocab_w2id = load_vocab_from_txt(args.vocab_path)
            else:
                print(
                    f"-- -- File does not have the required extension for loading the vocabulary. Exiting...")
                sys.exit()

            betas_path = model_config['betas_path']
            betas = np.load(betas_path)

        #  Get keys
        vocab_id2w = dict(zip(vocab_w2id.values(), vocab_w2id.keys()))
        keys = [
            [vocab_id2w[idx]
                for idx in row.argsort()[::-1][:top_words_display]]
            for row in betas
        ]
        all_topic_keys.append(keys)

        if model_config["remove_topic_ids"]:
            remove_topic_ids.append(
                [int(el) for el in model_config["remove_topic_ids"].split(",")])
        else:
            # append empty list to keep the order
            remove_topic_ids.append([])


    topic_selector = TopicSelector()
    selected_topics = topic_selector.iterative_matching(
        models=all_topic_keys, N=N, remove_topic_ids=remove_topic_ids)

    print(f"Selected topics: {selected_topics}")

    ############################
    # JSON formatter           #
    ############################
    method = config['all']['method']
    ntop = int(config['all']['ntop'])
    text_column = config['all']['text_column']
    text_column_disp = config['all']['text_column_disp']
    thr = config['all']['thr']
    thr = float(thr.split(",")[0]), float(thr.split(",")[1])
    path_json_save = config['all']['path_json_save']

    formatter = TopicJsonFormatter()

    idx_model = 0
    combined_out = {}
    for model in config.sections():
        model_config = config[model]
        if model != 'all':  # Skip all section (configuration for all models)
            model_config = config[model]
            print(f"-- -- Obtaining output for model {model}")
            # if trained with this repo code, we only need the model path
            if model_config.getboolean('trained_with_thetas_eval'):
                model_path = model_config['model_path']
                try:
                    # Load matrices
                    thetas = sparse.load_npz(pathlib.Path(
                        model_path) / "thetas.npz").toarray()
                    betas = np.load(pathlib.Path(model_path) / "betas.npy")

                    # Load vocab dictionaries
                    vocab_w2id = {}
                    with (pathlib.Path(model_path)/'vocab.txt').open('r', encoding='utf8') as fin:
                        for i, line in enumerate(fin):
                            wd = line.strip()
                            vocab_w2id[wd] = i

                except Exception as e:
                    print(
                        f"-- -- Error occurred when loading info from model {model_path.as_posix(): e}")
            else:
                model_path = None
                thetas_path = model_config['thetas_path']
                betas_path = model_config['betas_path']
                vocab_path = model_config['vocab_path']

                # Load matrices
                thetas = np.load(thetas_path)
                betas = np.load(betas_path)

                if vocab_path.endswith(".json"):
                    with open(vocab_path) as infile:
                        vocab_w2id = json.load(infile)

                elif vocab_path.endswith()(".txt"):
                    vocab_w2id = load_vocab_from_txt(args.vocab_path)
                else:
                    print(
                        f"-- -- File does not have the required extension for loading the vocabulary. Exiting...")
                    sys.exit()

            #  Get keys
            vocab_id2w = dict(zip(vocab_w2id.values(), vocab_w2id.keys()))
            keys = [
                [vocab_id2w[idx]
                    for idx in row.argsort()[::-1][:top_words_display]]
                for row in betas
            ]
            # print id and top words
            for i, key in enumerate(keys):
                print(f"-- -- Topic {i}: {key[:10]}")

            #  Get corpus
            corpus_path = pathlib.Path(model_config['corpus_path'])
            if corpus_path.suffix == ".parquet":
                df = pd.read_parquet(corpus_path)
            elif corpus_path.suffix in [".json", ".jsonl"]:
                df = pd.read_json(corpus_path, lines=True)
            else:
                print(
                    f"-- -- Unrecognized file extension for data path: {corpus_path.suffix}. Exiting...")
                sys.exit()

            df["text_split"] = df[text_column].apply(lambda x: x.split())
            corpus = df["text_split"].values.tolist()

            out = formatter.get_model_eval_output(
                df=df,
                text_column=text_column_disp,
                keys=keys,
                method=method,
                thetas=thetas,
                betas=betas,
                corpus=corpus,
                vocab_w2id=vocab_w2id,
                model_path=model_path,
                thr=thr,
                ntop=ntop)

            this_model_tpc_to_keep = []
            for pairs in selected_topics:
                for el in pairs:
                    if el[0] == idx_model:
                        this_model_tpc_to_keep.append(el[1])

            print(
                f"-- -- Topics to keep for model {model}: {this_model_tpc_to_keep}")

            # Filter JSON output
            filtered_out = {int(key): out[key]
                            for key in this_model_tpc_to_keep if key in out}

            if not model_path:
                model_path = pathlib.Path(thetas_path).parent.as_posix()
            print(
                f"-- -- Output for model {model} has {len(filtered_out)} topics")
            print(
                f"-- -- Saving output for model {model} with key {model_path}")

            combined_out.update({
                model_path: filtered_out
            })

            idx_model += 1

    # Write JSON output to file
    output_path = pathlib.Path(path_json_save) / \
        (pathlib.Path(args.config).stem + ".json")
    with open(output_path, 'w') as file:
        json.dump(combined_out, file, indent=4)

    # Return the JSON output
    # print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
