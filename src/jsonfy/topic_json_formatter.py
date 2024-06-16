import argparse
import json
import sys
from typing import List, Optional
import pathlib
import logging

import numpy as np
from scipy import sparse
from typing import List, Dict, Any
import pandas as pd
from src.topics_docs_selection.doc_selector import DocSelector
from src.utils.utils import init_logger


class TopicJsonFormatter:
    """
    Generates JSON output for topic models, including:

    - Representative Documents (methods: 'thetas', 'thetas_sample', 'thetas_thr', 'sall', 'spart', 's3')
    - Top Words for Each Topic
    - Evaluation Documents and Their Probabilities
    """

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

    def get_model_eval_output(
        self,
        df: pd.DataFrame,
        text_column: str,
        keys: List[str],
        **kwargs
    ) -> dict:
        """
        Get the model evaluation output in JSON format.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the text data.
        text_column : str
            Column name in the dataframe that contains the text data.
        keys : List[str]
            List of keys or topic words.
        kwargs : dict
            Additional keyword arguments for document selection.

        Returns
        -------
        dict
            JSON formatted dictionary of the model evaluation output.
        """
        exemplar_docs_ids = self._doc_selector.get_top_docs(**kwargs)
        exemplar_docs = [[df[text_column].iloc[doc_id]
                          for doc_id in k] for k in exemplar_docs_ids]
        eval_docs_ids, eval_docs_probs, assigned_to_k = self._doc_selector.get_eval_docs(
            exemplar_docs=exemplar_docs_ids,
            **kwargs)
        eval_docs = [[df[text_column].iloc[doc_id]
                      for doc_id in k] for k in eval_docs_ids]
        topic_ids = range(len(exemplar_docs_ids))
        
        distractor_ids = self._doc_selector.get_doc_distractor(**kwargs)
        distractor_docs = [df[text_column].iloc[doc_id] for doc_id in distractor_ids]
        
        json_out = self.jsonfy(topic_ids, exemplar_docs, keys,
                               eval_docs_ids, eval_docs, eval_docs_probs, distractor_ids, distractor_docs, assigned_to_k)
        return json_out

    def jsonfy(
        self,
        topic_ids: List[int],
        exemplar_docs: List[List[str]],
        topic_words: List[str],
        eval_docs_ids: List[List[int]],
        eval_docs: List[List[str]],
        eval_docs_probs: List[List[float]],
        distractor_docs_ids: List[List[int]],
        distractor_docs: List[List[str]],
        assigned_to_k: List[List[bool]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Convert model evaluation output to JSON format.

        Parameters
        ----------
        topic_ids : List[int]
            List of topic IDs.
        exemplar_docs : List[List[str]]
            List of exemplar documents for each topic.
        topic_words : List[str]
            List of topic words.
        eval_docs_ids : List[List[int]]
            List of evaluation document IDs for each topic.
        eval_docs : List[List[str]]
            List of evaluation documents for each topic.
        eval_docs_probs : List[List[float]]
            List of probabilities for the evaluation documents.
        distractor_docs_ids : List[List[int]]
            List with the distractor document ID for each topic.
        distractor_docs : List[List[str]]
            List with the distractor document for each topic.
        assigned_to_k : List[List[bool]]
            List of boolean values indicating document assignments.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            JSON formatted dictionary of the model evaluation output.
        """
        dict_out = {}

        for k, ed, tw, edids, evtext, edprobs, distid, disttext, edast in zip(
            topic_ids, exemplar_docs, topic_words, eval_docs_ids, eval_docs, eval_docs_probs, distractor_docs_ids, distractor_docs, assigned_to_k
        ):
            edast_b = [item == k for item in edast]

            eval_docs_list = [
                {
                    "doc_id": int(edids_d),
                    "text": evtext_d,
                    "prob": float(edprobs_d),
                    "assigned_to_k": bool(edast_b_d)
                }
                for edids_d, evtext_d, edprobs_d, edast_b_d in zip(edids, evtext, edprobs, edast_b)
            ]
            
            distractor_doc = {
                    "doc_id": int(distid),
                    "text": disttext
                }
                
            dict_out[int(k)] = {
                "exemplar_docs": ed,
                "topic_words": tw,
                "eval_docs": eval_docs_list,
                "distractor_doc": distractor_doc
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
    
    def display_json_snippet(json_obj, num_items=5, snippet_length=100, level=0, indent="  "):
        """
        Displays a snippet from each element in a JSON object, handling nested JSON objects.
        
        Parameters
        ----------
        json_obj: dict or list
            The JSON object to display snippets from.
        num_items: int
            Number of items to display from the JSON object.
        snippet_length: int
            Length of the snippet to display from each item.
        level: int
            Current level of nesting.
        indent: str
            String used for indentation.
        """
        current_indent = indent * level
        
        if isinstance(json_obj, dict):
            for i, (key, value) in enumerate(json_obj.items()):
                if i >= num_items:
                    break
                if isinstance(value, (dict, list)):
                    print(f"{current_indent}{key}:")
                    display_json_snippet(value, num_items, snippet_length, level + 1, indent)
                else:
                    value_snippet = str(value)[:snippet_length]
                    print(f"{current_indent}{key}: {value_snippet}")
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                if i >= num_items:
                    break
                if isinstance(item, (dict, list)):
                    print(f"{current_indent}Item {i + 1}:")
                    display_json_snippet(item, num_items, snippet_length, level + 1, indent)
                else:
                    item_snippet = str(item)[:snippet_length]
                    print(f"{current_indent}Item {i + 1}: {item_snippet}")


    def display_json_snippet(json_obj, num_items=5, snippet_length=100, level=0, indent="  "):
        """
        Displays a snippet from each element in a JSON object, handling nested JSON objects.

        Parameters
        ----------
        json_obj: dict or list
            The JSON object to display snippets from.
        num_items: int
            Number of items to display from the JSON object.
        snippet_length: int
            Length of the snippet to display from each item.
        level: int
            Current level of nesting.
        indent: str
            String used for indentation.
        """
        current_indent = indent * level

        if isinstance(json_obj, dict):
            for i, (key, value) in enumerate(json_obj.items()):
                if i >= num_items:
                    break
                if isinstance(value, (dict, list)):
                    print(f"{current_indent}{key}:")
                    display_json_snippet(
                        value, num_items, snippet_length, level + 1, indent)
                else:
                    value_snippet = str(value)[:snippet_length]
                    print(f"{current_indent}{key}: {value_snippet}")
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                if i >= num_items:
                    break
                if isinstance(item, (dict, list)):
                    print(f"{current_indent}Item {i + 1}:")
                    display_json_snippet(
                        item, num_items, snippet_length, level + 1, indent)
                else:
                    item_snippet = str(item)[:snippet_length]
                    print(f"{current_indent}Item {i + 1}: {item_snippet}")

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
        '--top_words_display',
        type=int,
        default=100,
        required=False,
        help='Number of top words per topic to display in the TopicJsonFormatter output.')
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
        action='store_true',
        help="Whether the model given by model_path was trained using this code"
    )
    parser.add_argument(
        '--text_column',
        type=str,
        required=False,
        default="tokenized_text",
        help='Column of corpus_path that was used for training the model.'
    )
    parser.add_argument(
        '--text_column_disp',
        type=str,
        required=False,
        default="text",
        help="Column data to display as document' contnet in TopicJsonFormatter."
    )

    args = parser.parse_args()

    formatter = TopicJsonFormatter()
    
    if args.trained_with_thetas_eval:
        model_path = pathlib.Path(args.model_path)
        try:
            # Load matrices
            thetas = sparse.load_npz(model_path / "thetas.npz").toarray()
            betas = np.load(model_path / "betas.npy")
            bow = sparse.load_npz(model_path / "bow.npz").toarray()

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
        # Load matrices
        thetas = np.load(args.thetas_path) if args.thetas_path else None
        try:
            bow = np.load(args.bow_path) if args.bow_path else None
        except Exception as e:
            # print(f"-- -- No BoW file found. Setting to None...")
            bow = None
        betas = np.load(args.betas_path) if args.betas_path else None

        if args.vocab_path.endswith(".json"):
            with open(args.vocab_path) as infile:
                vocab_w2id = json.load(infile)

        elif args.vocab_path.endswith()(".txt"):
            vocab_w2id = load_vocab_from_txt(args.vocab_path)
        else:
            print(
                f"-- -- File does not have the required extension for loading the vocabulary. Exiting...")
            sys.exit()

    #  Get keys
    vocab_id2w = dict(zip(vocab_w2id.values(), vocab_w2id.keys()))
    keys = [
        [vocab_id2w[idx] for idx in row.argsort()[::-1][:args.top_words_display]]
        for row in betas
    ]

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

    if args.thr:
        try:
            thr = float(args.thr.split(",")[0]), float(args.thr.split(",")[1])
        except Exception as e:
            print("-- -- The threshold values introduced are not valid. Please, introduce something in the format (inf_thr, sup_thr). Exiting...")
            sys.exit()

    out = formatter.get_model_eval_output(
        df=df,
        text_column=args.text_column_disp,
        keys=keys,
        method=args.method,
        thetas=thetas,
        bow=bow,
        betas=betas,
        corpus=corpus,
        vocab_w2id=vocab_w2id,
        model_path=args.model_path,
        top_words=args.top_words,
        thr=thr,
        ntop=args.ntop)

    # print("Keys:")
    # print(list(out.keys()))

    # print("\nSnippets from output:")
    # display_json_snippet(out, num_items=5, snippet_length=100)

    # Return the JSON output
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
