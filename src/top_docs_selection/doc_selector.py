import argparse
import json
import logging
import pathlib
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from typing import List, Tuple, Dict, Optional
from src.utils.utils import init_logger, keep_top_k_values


class DocSelector(object):
    """
    Class with different implementation to select the most representative documents per topic for a given topic model.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ) -> None:
        """
        Initialize the DocSelector class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger object to log activity.
        path_logs : pathlib.Path, optional
            Path for saving logs.
        """
        self._logger = logger if logger else init_logger(__name__, path_logs)

        return

    def _get_words_assigned(
        self,
        bow: np.ndarray,
        thetas: np.ndarray,
        betas: np.ndarray,
        doc_id: int,
        tp_id: int
    ) -> List[int]:
        """
        Simulate LDA's word assignment process for a given document and topic.

        Parameters
        ----------
        bow : np.ndarray
            Bag of words representation of the documents.
        thetas : np.ndarray
            Document-topic distributions.
        betas : np.ndarray
            Topic-word distributions.
        doc_id : int
            Document ID for which words are to be assigned.
        tp_id : int
            Topic ID for which words are to be assigned.

        Returns
        -------
        list
            List of word indices assigned to the specified topic in the given document.
        """
        words_doc_idx = [i for i, val in enumerate(bow[doc_id]) if val > 0]
        thetas_doc = thetas[doc_id]

        words_assigned = []
        for idx_w in words_doc_idx:
            p_z = np.multiply(thetas_doc, betas[:, idx_w])
            p_z_args = np.argsort(p_z)
            if p_z[p_z_args[-1]] > 20 * p_z[p_z_args[-2]]:
                assignment = p_z_args[-1]
                if assignment == tp_id:
                    words_assigned.append(idx_w)
            else:
                sampling = np.random.multinomial(1, np.multiply(
                    thetas_doc, betas[:, idx_w]) / np.sum(np.multiply(thetas_doc, betas[:, idx_w])))
                assignment = int(np.nonzero(sampling)[0][0])
                if assignment == tp_id:
                    words_assigned.append(idx_w)

        return words_assigned

    def _calculate_bhata(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calculate the Bhattacharyya distance between two distributions.

        Parameters
        ----------
        X : np.ndarray
            First distribution.
        Y : np.ndarray
            Second distribution.

        Returns
        -------
        np.ndarray
            Bhattacharyya distance between X and Y.
        """
        try:
            if len(X) == 2 and len(Y) == 2:
                return np.sqrt(X) * np.sqrt(Y.T)
            else:
                return np.sum(X * Y)
        except:
            if len(X.shape) == 2 and len(Y.shape) == 2:
                return sparse.csr_matrix(np.sqrt(X) * np.sqrt(Y.T))
            else:
                return np.sum(X.multiply(Y))

    def _calculate_sall(
        self,
        bow: np.ndarray,
        betas: np.ndarray,
        save_path: Optional[str] = None
    ) -> sparse.csr_matrix:
        """
        Calculate Sall (similarities between BoW and betas).

        Parameters
        ----------
        bow : np.ndarray
            Bag of words matrix.
        betas : np.ndarray
            Topic-word distributions.
        save_path : str, optional
            Path to save the results.

        Returns
        -------
        sparse.csr_matrix
            Similarity matrix between BoW and betas.
        """

        if save_path:
            save_path = pathlib.Path(save_path)

        try:
            S_all = sparse.load_npz(save_path / 'top_docs_out/S_all.npz')
            self._logger.info(f"-- -- S_all loaded from file.")
            return S_all
        except Exception as e:
            self._logger.warning(
                f"-- -- S_all could not be loaded from file: {e}. It will be calculated.")

        self._logger.info(
            f"-- -- Calculating Sall (similarities between BoW and betas)...")
        start = time.time()

        bow_mat_norm = bow.copy()
        row_sums = bow_mat_norm.sum(axis=1)
        bow_mat_norm = bow_mat_norm / row_sums[:, np.newaxis]
        bow_mat_sparse = sparse.csr_matrix(bow_mat_norm)
        betas_sparse = sparse.csr_matrix(betas)
        S_all = np.sqrt(bow_mat_sparse) * np.sqrt(betas_sparse.T)

        self._logger.info(f"-- -- S_all shape: {S_all.toarray().shape}")
        self._logger.info(f"-- -- Time elapsed: {time.time() - start}")

        if save_path:
            if not (save_path / "top_docs_out").exists():
                (save_path / "top_docs_out").mkdir(exist_ok=True)
            sparse.save_npz(save_path.joinpath(
                'top_docs_out/S_all.npz'), S_all)

        return S_all

    def _calculate_spart(
        self,
        bow: np.ndarray,
        thetas: np.ndarray,
        betas: np.ndarray,
        save_path: Optional[str] = None
    ) -> sparse.csr_matrix:
        """
        Calculate Spart (similarities between BoW particularized and betas for each document-topic pair).

        Parameters
        ----------
        bow : np.ndarray
            Bag of words matrix.
        thetas : np.ndarray
            Document-topic distributions.
        betas : np.ndarray
            Topic-word distributions.
        save_path : str, optional
            Path to save the results.

        Returns
        -------
        sparse.csr_matrix
            Similarity matrix for each document-topic pair.
        """

        if save_path:
            save_path = pathlib.Path(save_path)

        try:
            S_part_sparse = sparse.load_npz(
                save_path / 'top_docs_out/S_part.npz')
            self._logger.info(f"-- -- Spart loaded from file.")
            return S_part_sparse
        except Exception as e:
            self._logger.warning(
                f"-- -- Spart could not be loaded from file: {e}. It will be calculated.")

        self._logger.info(
            f"-- -- Calculating Spart (similarities between BoW particularized and betas for each document-topic pair)...")
        start = time.time()

        S_part = np.zeros((len(thetas), len(betas)))
        for doc in range(len(thetas)):
            for topic in range(thetas.shape[1]):
                words_assigned = self._get_words_assigned(
                    bow, thetas, betas, doc, topic)
                mask = np.ones(bow.shape[1], dtype=bool)
                mask[words_assigned] = False
                bow_doc = bow[doc].flatten()
                bow_doc[mask] = 0
                if np.any(bow_doc != 0):
                    bow_doc = bow_doc / np.linalg.norm(bow_doc)
                sim = self._calculate_bhata(bow_doc, betas[topic])
                S_part[doc, topic] = sim

        self._logger.info(f"-- -- S_part shape: {S_part.shape}")
        self._logger.info(f"-- -- Time elapsed: {time.time() - start}")

        S_part_sparse = sparse.csr_matrix(S_part)

        if save_path:
            if not (save_path / "top_docs_out").exists():
                (save_path / "top_docs_out").mkdir(exist_ok=True)
            sparse.save_npz(save_path.joinpath(
                'top_docs_out/S_part.npz'), S_part_sparse)

        return S_part_sparse

    def _calculate_s3(
        self,
        thetas: np.ndarray,
        betas: np.ndarray,
        corpus: List[List[str]],
        vocab_w2id: dict,
        top_words: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> sparse.csr_matrix:
        """
        Calculate S3 (sum of the weights that topic assigns to each word in the document).

        Parameters
        ----------
        thetas : np.ndarray
            Document-topic distributions.
        betas : np.ndarray
            Topic-word distributions.
        corpus : List[List[str]]
            List of documents, each document being a list of words.
        vocab_w2id : dict
            Dictionary mapping words to their indices in the vocabulary.
        top_words : int, optional
            Number of top words to keep in the betas matrix, by default None.
        save_path : str, optional
            Path to save the results.

        Returns
        -------
        sparse.csr_matrix
            Similarity matrix based on topic weights assigned to words in the documents.
        """

        if save_path:
            save_path = pathlib.Path(save_path)

        try:
            S3_sparse = sparse.load_npz(save_path / 'top_docs_out/S3.npz')
            self._logger.info(f"-- -- S3 loaded from file.")
            return S3_sparse
        except Exception as e:
            self._logger.warning(
                f"-- -- S3 could not be loaded from file: {e}. It will be calculated.")

        self._logger.info(
            f"-- -- Calculating S3 (sum of the weights that topic assigns to each word in the document)...")
        start = time.time()

        S3 = np.zeros((len(thetas), len(betas)))

        betas_top = keep_top_k_values(betas, top_words) if top_words else betas

        for doc in range(len(thetas)):
            for topic in range(thetas.shape[1]):
                wd_ids = [vocab_w2id.get(word)
                          for word in corpus[doc] if word in vocab_w2id]
                S3[doc, topic] = np.sum(betas_top[topic, wd_ids])

        self._logger.info(f"-- -- S3 shape: {S3.shape}")
        self._logger.info(f"-- -- Time elapsed: {time.time() - start}")

        S3_sparse = sparse.csr_matrix(S3)

        if save_path:
            if not (save_path / "top_docs_out").exists():
                (save_path / "top_docs_out").mkdir(exist_ok=True)
            sparse.save_npz(save_path.joinpath(
                'top_docs_out/S3.npz'), S3_sparse)

        return S3_sparse

    def _get_most_representative_per_tpc(
        self,
        mat: np.ndarray,
        topn: int = 10
    ) -> List[List[int]]:
        """
        Finds the most representative documents for each topic based on a given matrix.

        Parameters
        ----------
        mat: numpy.ndarray
            The input matrix of shape (D, K) where D is the number of documents and K is the number of topics. Each element represents the distribution of a document over a topic.
        topn: int, optional
            The number of top documents to select for each topic. Defaults to 10.

        Returns
        -------
        List[List[int]]: 
            A list of lists containing the indices of the most representative documents for each topic.
        """
        # Find the most representative document for each topic based on a matrix mat
        top_docs_per_topic = []

        for doc_distr in mat.T:
            sorted_docs_indices = np.argsort(doc_distr)[::-1]
            top = sorted_docs_indices[:topn].tolist()
            top_docs_per_topic.append(top)

        return top_docs_per_topic

    def _select_ids_nparts(
        self,
        mat: np.ndarray,
        n_parts: int = 5
    ) -> List[List[int]]:
        """
        Selects one random value from each of the n_parts segments of each column in the given matrix. If the number of rows is not evenly divisible by n_parts, the last segment will include the remaining elements.

        Parameters
        ----------
        mat: numpy.ndarray
            The input matrix of shape (D, K) where D is the number of rows and K is the number of columns.
        n_parts: int
            The number of segments to divide each column into.

        Returns
        -------
        List[List[int]]: 
            A list of lists containing the indices of the selected values for each column. Each inner list corresponds to a column and contains one selected index from each of the n_parts segments.
        """
        selected_ids = []

        for col in range(mat.shape[1]):
            column_data = mat[:, col]
            # Sort in descending order
            sorted_indices = np.argsort(-column_data)
            sorted_data = column_data[sorted_indices]

            part_size = len(sorted_data) // n_parts
            this_col_ids = []

            # Select one value from each part in sorted data
            for i in range(n_parts):
                start_index = i * part_size
                end_index = (i + 1) * part_size if i < n_parts - \
                    1 else len(sorted_data)
                part_indices = sorted_indices[start_index:end_index]
                if part_indices.size > 0:
                    selected_index = np.random.choice(part_indices)
                    this_col_ids.append(selected_index)

            # Append the selected indices as a list to the result list
            selected_ids.append(this_col_ids)

        return selected_ids

    def get_top_docs(
            self,
            method: str,
            thetas: np.ndarray = None,
            bow: np.ndarray = None,
            betas: np.ndarray = None,
            corpus: List[List[str]] = None,
            vocab_w2id: dict = None,
            model_path: str = None,
            top_words: int = None,
            thr: tuple = None,
            ntop: int = 5) -> List[List[int]]:
        """
        Get the top documents based on the specified method.

        Parameters
        ----------
        method : str
            Method to use for selecting top documents.
        model_path : str
            Path to the model.
        ntop : int, default=5
            Number of top documents to select.
        """

        if method == "thetas":
            mat = thetas.copy()
        elif method == "thetas_sample":
            mat = thetas.copy()
            mat = mat / mat.sum(axis=0)
            top_docs = []
            for col in range(len(mat.T)):
                top_docs_per_topic = []
                for _ in range(ntop):
                    sampled_idx = np.random.choice(len(mat), p=mat[:, col])
                    top_docs_per_topic.append(sampled_idx)
                top_docs.append(top_docs_per_topic)
            return top_docs
        elif method == "thetas_thr":
            mat = thetas.copy()
            mask = (mat > thr[0]) & (mat < thr[1])
            mat = np.where(mask, mat, 0)
        elif method == "sall":
            mat = self._calculate_sall(
                bow, betas, save_path=model_path).toarray()
        elif method == "spart":
            mat = self._calculate_spart(
                bow, thetas, betas, save_path=model_path).toarray()
        elif method == "s3":
            mat = self._calculate_s3(
                thetas, betas, corpus, vocab_w2id, top_words, save_path=model_path).toarray()

        return self._get_most_representative_per_tpc(mat, ntop)

    def get_eval_docs(
        self,
        method: str,
        thetas: np.ndarray = None,
        bow: np.ndarray = None,
        betas: np.ndarray = None,
        corpus: List[List[str]] = None,
        vocab_w2id: Dict[str, int] = None,
        model_path: str = None,
        top_words: int = None,
        thr: Tuple[float, float] = None,
        ntop: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Get the documents for evaluation. If we want N evaluation docs, we divide each column of the D x K matrix into N parts and select one element from each part. If the selected method is 'thetas_sample', the eval docs are selected as for the 'thetas' method.

        Parameters
        ----------
        method : str
            Method to use for selecting top documents.
        thetas : np.ndarray, optional
            Topic proportions for documents.
        bow : np.ndarray, optional
            Bag-of-words representation of the corpus.
        betas : np.ndarray, optional
            Topic-word distributions.
        corpus : List[List[str]], optional
            The corpus as a list of lists of words.
        vocab_w2id : dict, optional
            Vocabulary word-to-id mapping.
        model_path : str, optional
            Path to the model.
        top_words : int, optional
            Number of top words to consider.
        thr : tuple, optional
            Threshold values for filtering topics.
        ntop : int, default=5
            Number of top documents to select.

        Returns
        -------
        Tuple[List[List[int]], List[List[float]]]: 
            A tuple containing a list of lists of the IDs of the selected documents for each topic and their corresponding probabilities.
        """

        if method == "thetas" or method == "thetas_sample":
            mat = thetas.copy()
        elif method == "thetas_thr":
            mat = thetas.copy()
            mask = (mat > thr[0]) & (mat < thr[1])
            mat = np.where(mask, mat, 0)
        elif method == "sall":
            mat = self._calculate_sall(
                bow, betas, save_path=model_path).toarray()
        elif method == "spart":
            mat = self._calculate_spart(
                bow, thetas, betas, save_path=model_path).toarray()
        elif method == "s3":
            mat = self._calculate_s3(
                thetas, betas, corpus, vocab_w2id, top_words, save_path=model_path).toarray()

        eval_docs = self._select_ids_nparts(mat, ntop)
        eval_probs = [[thetas.T[k][doc_id] for doc_id in id_docs]
                      for k, id_docs in enumerate(eval_docs)]

        return eval_docs, eval_probs


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

    doc_selector = DocSelector()

    if args.trained_with_thetas_eval:
        model_path = pathlib.Path(args.model_path)
        try:
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
        top_docs = doc_selector.get_top_docs(
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

        print("Top documents selected using method:", method)
        for topic_idx, docs in enumerate(top_docs):
            print(f"Topic {topic_idx}: {docs}")


if __name__ == "__main__":
    main()
