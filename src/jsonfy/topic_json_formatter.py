import argparse
import json
import sys
from typing import List, Optional
import pathlib
import logging

import numpy as np
from scipy import sparse
import pandas as pd
from src.top_docs_selection.doc_selector import DocSelector
from src.utils.utils import init_logger


class TopicJsonFormatter(object):
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ) -> None:
        """
        Initialize the TopicJsonFormatter class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger object to log activity.
        path_logs : pathlib.Path, optional
            Path for saving logs.
        """
        self._logger = logger if logger else init_logger(__name__, path_logs)

        self._doc_selector = DocSelector(self._logger, path_logs)

        return

    def get_model_eval_output(self, df, text_column, keys, **kwargs):

        exemplar_docs_ids = self._doc_selector.get_top_docs(**kwargs)
        exemplar_docs = [[df[text_column].iloc[doc_id]
                          for doc_id in k] for k in exemplar_docs_ids]
        eval_docs_ids, eval_docs_probs = self._doc_selector.get_eval_docs(
            **kwargs)
        eval_docs = [[df[text_column].iloc[doc_id]
                      for doc_id in k] for k in eval_docs_ids]
        topic_ids = range(len(exemplar_docs_ids))

        json_out = self.jsonfy(topic_ids, exemplar_docs,
                               keys, eval_docs, eval_docs_probs)
        
        import pdb; pdb.set_trace()
        return json_out

    def jsonfy(
        self,
        topic_ids,
        exemplar_docs,
        topic_words,
        eval_docs,
        eval_docs_probs,
        path_save=None
    ):
        dict_out = {}
        for k, ed, tw, ed, edp in zip(topic_ids, exemplar_docs, topic_words, eval_docs, eval_docs_probs):
            dict_out[k] = {
                "exemplar_docs": ed,
                "topic_words": tw,
                "eval_docs": ed,
                "eval_docs_probs": edp
            }
        return dict_out


def main():

    def load_vocab_from_txt(vocab_file):
        vocab_w2id = {}
        with (vocab_file).open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                wd = line.strip()
                vocab_w2id[wd] = i
        return vocab_w2id

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        help="Method to use for selecting top documents. Several methods can be specified by providing them as a string separated by commas. Available methods: 'thetas', 'thetas_sample', 'thetas_thr', 'sall', 'spart', 's3'")
    parser.add_argument(
        '--thetas_path',
        type=str,
        required=False,
        help='Path to the thetas numpy file.'
    )
    parser.add_argument(
        '--bow_path',
        type=str,
        required=False,
        help='Path to the bag-of-words numpy file.')
    parser.add_argument(
        '--betas_path',
        type=str,
        required=False,
        help='Path to the betas numpy file.'
    )
    parser.add_argument(
        '--corpus_path',
        type=str,
        required=False,
        help='Path to the corpus file.'
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        required=False,
        help='Path to the vocabulary file (word to index mapping).'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=False,
        help='Path to the model directory.'
    )
    parser.add_argument(
        '--top_words',
        type=int,
        required=False,
        help='Number of top words to keep in the betas matrix when using S3.')
    parser.add_argument(
        '--thr',
        type=str,
        required=False,
        default="0.1,0.8",
        help='Threshold values for the thetas_thr method, as (thr_inf,thr_sup)'
    )
    parser.add_argument(
        '--ntop',
        type=int,
        default=5,
        help='Number of top documents to select.'
    )
    parser.add_argument(
        '--trained_with_thetas_eval',
        type=bool,
        default=True,
        help="Whether the model given by model_path was trained using this code"
    )
    parser.add_argument(
        '--text_column',
        type=str,
        required=False,
        default="tokenized_text",
        help='Column of corpus_path that was used for training the model.'
    )

    args = parser.parse_args()

    formatter = TopicJsonFormatter()

    if args.trained_with_thetas_eval:
        model_path = pathlib.Path(args.model_path)
        try:
            thetas = sparse.load_npz(model_path / "thetas.npz").toarray()
            betas = np.load(model_path / "betas.npy")
            bow = sparse.load_npz(model_path / "bow.npz").toarray()
            with (model_path / "tpc_descriptions.txt").open('r', encoding='utf8') as fin:
                keys = [el.strip() for el in fin.readlines()]

            # Load vocab dictionaries
            vocab_w2id = {}
            with (model_path/'vocab.txt').open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    wd = line.strip()
                    vocab_w2id[wd] = i

        except Exception as e:
            print(
                f"-- -- Error occurred when loading info from model {model_path.as_posix(): e}")

    else:
        thetas = np.load(args.thetas_path) if args.thetas_path else None
        bow = np.load(args.bow_path) if args.bow_path else None
        betas = np.load(args.betas_path) if args.betas_path else None

        if args.vocab_path.endswith(".json"):
            with open(args.vocab_path) as infile:
                vocab_w2id = json.load(infile)
            # vocab_id2w = dict(zip(vocab_w2id.values(), vocab_w2id.keys()))
        elif args.vocab_path.endswith()(".txt"):
            vocab_w2id = load_vocab_from_txt(args.vocab_path)
        else:
            print(
                f"-- -- File does not have the required extension for loading the vocabulary. Exiting...")
            sys.exit()

    corpus_path = pathlib.Path(args.corpus_path)
    if corpus_path.suffix == ".parquet":
        df = pd.read_parquet(corpus_path)
    elif corpus_path.suffix in [".json", ".jsonl"]:
        df = pd.read_json(corpus_path, lines=True)
    else:
        print(
            f"-- -- Unrecognized file extension for data path: {corpus_path.suffix}. Exiting...")
        sys.exit()

    df["text_split"] = df[args.text_column].apply(lambda x: x.split())
    corpus = df["text_split"].values.tolist()

    try:
        methods = args.method.split(",")
    except:
        methods = [args.method]

    if args.thr:
        try:
            thr = float(args.thr.split(",")[0]), float(args.thr.split(",")[1])
        except Exception as e:
            print("The threshold values introduced are not valid. Please, introduce something in the format (inf_thr, sup_thr). Exiting...")
            sys.exit()

    for method in methods:
        print(f"Getting with {method}")
        formatter.get_model_eval_output(
            df=df,
            text_column=args.text_column,
            keys=keys,
            method=method,
            thetas=thetas,
            bow=bow,
            betas=betas,
            corpus=corpus,
            vocab_w2id=vocab_w2id,
            model_path=args.model_path,
            top_words=args.top_words,
            thr=thr,
            ntop=args.ntop)

if __name__ == "__main__":
    main()
