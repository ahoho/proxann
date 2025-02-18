import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy import sparse
from scipy.ndimage import uniform_filter1d

from src.utils.utils import init_logger, keep_top_k_values, load_vocab_from_txt, safe_load_npy


class DocSelector(object):
    """
    Class with different approaches to select the most representative documents per topic for a given topic model.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent / "config/config.yaml"
    ) -> None:
        """
        Initialize the DocSelector class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger object, by default None
        config_path : pathlib.Path, optional
            Path to the configuration file, by default pathlib.Path(__file__).parent.parent.parent / "config/config.yaml"
        """
        self._logger = logger if logger else init_logger(config_path,__name__)

        return

    def _get_assign_tpc(self, thetas):
        """Get the topic assignment for each document based on the document-topic distributions as the most probable topic.

        Parameters
        ----------
        thetas : np.ndarray
            Document-topic distributions.

        Returns
        -------
        np.ndarray
            Topic assignments for each document.
        """
        return np.argmax(thetas, axis=1)

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

    def _calculate_bhata(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
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

        betas_top = keep_top_k_values(betas, top_words, self._logger) if top_words else betas

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
        Finds the most representative documents for each topic based on a given matrix (e.g., thetas, s3, etc.)

        Parameters
        ----------
        mat: numpy.ndarray
            The input matrix of shape (D, K) where D is the number of documents and K is the number of topics. 
        topn: int, optional
            The number of top documents to select for each topic. Defaults to 10.

        Returns
        -------
        List[List[int]]: 
            A list of lists containing the indices of the most representative documents for each topic.
        """
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
        Selects one random value from each of the n_parts segments of each column in the given matrix.
        If the number of rows is not evenly divisible by n_parts, the last segment will include the remaining elements.

        Parameters
        ----------
        mat: numpy.ndarray
            The input matrix of shape (D, K) where D is the number of documents and K is the number of topics.
        n_parts: int
            The number of segments to divide each column (probabilities of each document for a given column = topic) into.

        Returns
        -------
        List[List[int]]:
            A list of lists containing the indices of the selected documents for each topic. Each inner list corresponds to a topic and contains one selected document from each of the n_parts segments.
        """
        selected_ids = []
        mat = mat.copy()

        num_docs = mat.shape[0]
        if np.any(mat > 1):
            self._logger.info("Subtracting thetas > 1 (should only happen for the labeled docs)")
            mat[mat > 1] -=1
        if num_docs < n_parts:
            raise ValueError(f"Number of documents ({num_docs}) is less than the number of parts ({n_parts}).")

        for col in range(mat.shape[1]):
            column_data = mat[:, col]
            this_col_ids = []  # Initialize this_col_ids here
            
            max_mat_k = column_data.max()
            if max_mat_k == 0:
                step = 1e-10  # Avoid division by zero
            else:
                step = max_mat_k / (n_parts - 1)
            
            try:
                for p in np.arange(max_mat_k, -step, -step):
                    idx = np.abs(column_data - p).argmin()
                    this_col_ids.append(idx)
                    column_data[idx] = 1e10  # Exclude from future selection
                    if len(this_col_ids) == n_parts:
                        break
            except Exception as e:
                self._logger.error(f"Error occurred when bucketing by closest value: {e}")
                return

            # Fallback to ensure n_parts values
            if len(this_col_ids) < n_parts:
                self._logger.warning(f"-- -- Not enough unique values to select {n_parts} parts for column {col}. Fallback to random selection.")
                remaining_indices = np.where(0 <= column_data < 1e10)[0]
                np.random.shuffle(remaining_indices)
                for idx in remaining_indices:
                    if idx not in this_col_ids:
                        this_col_ids.append(idx)
                        if len(this_col_ids) == n_parts:
                            break


            # Sort such that probs are in descending order
            this_col_ids = sorted(this_col_ids, key=lambda idx: column_data[idx], reverse=True)
            
            selected_ids.append(this_col_ids)

        return selected_ids

    def _get_mat_for_top(
        self,
        method: str,
        thetas: np.ndarray = None,
        bow: np.ndarray = None,
        betas: np.ndarray = None,
        corpus: List[List[str]] = None,
        vocab_w2id: dict = None,
        model_path: str = None,
        top_words: int = None,
        thr: tuple = None
    ) -> List[List[int]]:
        """
        Calculate the matrix that is going to be used for selecing the top documents based on the specified method:
        For each topic, keep:
        * thetas: 
            the ntop-documents with the highest thetas (doc-topic distrib)
        * thetas_thr: 
            the ntop-documents are selected based on a threshold. The thetas matrix is filtered such that only values within the specified threshold range are kept.
        * sall: 
            Top docs are selected based on the largest Bhattacharya    coefficient between  their normalized BoW and the betas.
        * spart: 
            Top docs are chosen by identifying those with the largest Bhattacharya coefficient between the BoW of the document, specific to the words generated for the topic, and the topic's betas.
        * s3: 
            For each topic, top docs are chosen by keeping those with the largest sum of the  eights that such a topic assigns to each word in the document.

        Parameters
        ----------
        method : str
            Method to use for selecting top documents.
        exemplar_docs : List[int]
            List of IDs of the exemplar documents.
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
        np.ndarray:
            The matrix to be used for selecting the top documents.
        """
        if method == "thetas":
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

        return mat

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
        ntop: int = 5,
        poly_degree=3,
        smoothing_window=5
    ) -> List[List[int]]:
        """
        Get the top documents based on the specified method.
        For each topic, keep:
        * thetas: 
            the ntop-documents with the highest thetas (doc-topic distrib)
        * thetas_sample: 
            the ntop-documents are selected based on probabilistic sampling. The thetas matrix is normalized such that the columns sum to 1. For each topic, documents are sampled according to their probabilities in the normalized thetas matrix.
        * thetas_thr: 
            the ntop-documents are selected based on a threshold. The thetas matrix is filtered such that only values within the specified threshold range are kept.
        * elbow:
            The elbow point is calculated for each topic and ntop samples are selected based on the probabilities of the thetas matrix after filtering the values below the elbow point.
        * sall: 
            Top docs are selected based on the largest Bhattacharya    coefficient between  their normalized BoW and the betas.
        * spart: 
            Top docs are chosen by identifying those with the largest Bhattacharya coefficient between the BoW of the document, specific to the words generated for the topic, and the topic's betas.
        * s3: 
            For each topic, top docs are chosen by keeping those with the largest sum of the  eights that such a topic assigns to each word in the document.
        
        Parameters
        ----------
        method : str
            Method to use for selecting top documents.
        exemplar_docs : List[int]
            List of IDs of the exemplar documents.
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
        List[List[int]]:
            A list of lists containing the indices of the ntop selected documents for each topic.
        """

        if method == "thetas_sample":
            mat = thetas.copy()
            mat = mat / mat.sum(axis=0)
            most_representative_per_tpc = []
            for col in range(len(mat.T)):
                top_docs_per_topic = []
                for _ in range(ntop):
                    sampled_idx = np.random.choice(len(mat), p=mat[:, col])
                    top_docs_per_topic.append((sampled_idx, mat[sampled_idx, col]))
                # Sort by the probs in descending order and append just idxs
                top_docs_per_topic.sort(key=lambda x: x[1], reverse=True)
                most_representative_per_tpc.append([doc[0] for doc in top_docs_per_topic])
        
        elif method == "elbow":
            mat = thetas.copy()
            most_representative_per_tpc = []
            for k in range(len(mat.T)):
                allvalues = np.sort(mat[:, k].flatten())
                step = int(np.round(len(allvalues) / 1000))
                x_values = allvalues[::step]
                y_values = (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step]

                # Apply smoothing
                y_values_smooth = uniform_filter1d(y_values, size=smoothing_window)
                                
                # Find the elbow point
                kneedle = KneeLocator(x_values, y_values_smooth, curve='convex', direction='increasing', interp_method='polynomial', polynomial_degree=poly_degree)
                elbow = kneedle.elbow
                
                if elbow:
                    # Filter document indices based on the elbow (keeping values above the elbow)
                    significant_idx = np.where(mat[:, k] >= elbow)[0]
                    significant_values = mat[significant_idx, k]

                    # Normalize to get probability distribution and sample
                    probabilities = significant_values / np.sum(significant_values)

                    if len(significant_idx) > 0:
                        sampled_indices = np.random.choice(significant_idx, size=min(ntop, len(significant_idx)), p=probabilities, replace=False)
                        
                        # Sort by the probs in descending order and append idxs
                        sampled_indices = sorted(sampled_indices, key=lambda idx: mat[idx, k], reverse=True)
                        most_representative_per_tpc.append(sampled_indices)
                    else:
                        self._logger.warning(f"-- -- No documents found above the elbow point for topic {k}. Using thetas...")
                        sorted_docs_indices = np.argsort(mat.T[k])[::-1]
                        top = sorted_docs_indices[:ntop].tolist()
                        most_representative_per_tpc.append(top)
                    
                else:
                    self._logger.warning(f"-- -- No elbow point found for topic {k}. Using thetas...")
                    
                    sorted_docs_indices = np.argsort(mat.T[k])[::-1]
                    top = sorted_docs_indices[:ntop].tolist()
                    most_representative_per_tpc.append(top)
        else:
            mat = self._get_mat_for_top(
                method, thetas, bow, betas, corpus, vocab_w2id, model_path, top_words, thr)
            
            most_representative_per_tpc = self._get_most_representative_per_tpc(mat, ntop)

        exemplar_docs_probs = [[thetas.T[k][doc_id] for doc_id in id_docs]
            for k, id_docs in enumerate(most_representative_per_tpc)]
            
        return most_representative_per_tpc, exemplar_docs_probs
        
    def get_eval_docs(
        self,
        method: str,
        exemplar_docs: List[int],
        thetas: np.ndarray = None,
        bow: np.ndarray = None,
        betas: np.ndarray = None,
        corpus: List[List[str]] = None,
        vocab_w2id: Dict[str, int] = None,
        model_path: str = None,
        top_words: int = None,
        thr: Tuple[float, float] = None,
        ntop: int = 7
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Get the documents for evaluation. If we want N evaluation docs, we divide each column of the D x K matrix into N parts and select one element from each part. If the selected method is 'thetas_sample', the eval docs are selected as for the 'thetas' method.

        Parameters
        ----------
        method : str
            Method to use for selecting top documents.
        exemplar_docs : List[int]
            List of IDs of the exemplar documents.
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

        # Modify thetas to remove the documents that have already been selected as exemplar docs
        thetas_ = thetas.copy()
        thetas_[exemplar_docs] = -1

        if method == "thetas_sample" or method == "elbow":
            method = "thetas"

        mat = self._get_mat_for_top(
            method, thetas_, bow, betas, corpus, vocab_w2id, model_path, top_words, thr)

        eval_docs = self._select_ids_nparts(mat, ntop)
        eval_probs = [[thetas_.T[k][doc_id] for doc_id in id_docs]
                      for k, id_docs in enumerate(eval_docs)]
        
        try:
            assigned_to_k = self._get_assign_tpc(thetas_)[eval_docs]
        except Exception as e:
            self._logger.error(f"Error occurred when assigning topics to evaluation docs: {e}")
            assigned_to_k = None
        return eval_docs, eval_probs, assigned_to_k

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config_path",
        help="Path to the configuration file.",
        type=str,
        default="config/config.yaml",
        required=False
    )
    argparser.add_argument(
        '--method',
        type=str,
        required=True,
        default="elbow",
        help="Method to use for selecting top documents. Several methods can be specified by providing them as a string separated by commas. Available methods: 'thetas', 'thetas_sample', 'thetas_thr', 'sall', 'spart', 's3', 'elbow'.")
    argparser.add_argument(
        '--thetas_path',
        type=str,
        required=False,
        default=None,
        help='Path to the thetas numpy file.'
    )
    argparser.add_argument(
        '--bow_path',
        type=str,
        required=False,
        default=None,
        help='Path to the bag-of-words numpy file.')
    argparser.add_argument(
        '--betas_path',
        type=str,
        required=False,
        default=None,
        help='Path to the betas numpy file.'
    )
    argparser.add_argument(
        '--corpus_path',
        type=str,
        required=False,
        default=None,
        help='Path to the corpus file.'
    )
    argparser.add_argument(
        '--vocab_path',
        type=str,
        required=False,
        default=None,
        help='Path to the vocabulary file (word to index mapping).'
    )
    argparser.add_argument(
        '--model_path',
        type=str,
        required=False,
        default=None,
        help='Path to the model directory.'
    )
    argparser.add_argument(
        '--top_words',
        type=int,
        required=False,
        default=15,
        help='Number of top words to keep in the betas matrix when using S3.')
    argparser.add_argument(
        '--thr',
        type=str,
        required=False,
        default="0.1,0.8",
        help='Threshold values for the thetas_thr method, as (thr_inf,thr_sup)'
    )
    argparser.add_argument(
        '--ntop',
        type=int,
        default=5,
        help='Number of top documents to select.'
    )
    argparser.add_argument(
        '--trained_with_thetas_eval',
        action='store_true',
        help="Whether the model given by model_path was trained using this code"
    )
    argparser.add_argument(
        '--text_column',
        type=str,
        required=False,
        default="tokenized_text",
        help='Column of corpus_path that was used for training the model.'
    )

    args = argparser.parse_args()

    # Initialize the logger
    logger = init_logger(args.config_path, DocSelector.__name__)

    doc_selector = DocSelector(logger=logger)

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
            logger.info(
                f"-- -- Error occurred when loading info from model {model_path.as_posix(): e}")

    else:
        thetas = safe_load_npy(args.thetas_path, logger, "Thetas numpy file")
        bow = safe_load_npy(args.bow_path, logger, "Bag-of-words numpy file")
        betas = safe_load_npy(args.betas_path, logger, "Betas numpy file")

        if args.vocab_path.endswith(".json"):
            with open(args.vocab_path) as infile:
                vocab_w2id = json.load(infile)
            # vocab_id2w = dict(zip(vocab_w2id.values(), vocab_w2id.keys()))
        elif args.vocab_path.endswith(".txt"):
            vocab_w2id = load_vocab_from_txt(args.vocab_path)
        else:
            logger.info(
                f"-- -- File does not have the required extension for loading the vocabulary. Exiting...")
            sys.exit()

    corpus_path = pathlib.Path(args.corpus_path)
    if corpus_path.suffix == ".parquet":
        df = pd.read_parquet(corpus_path)
    elif corpus_path.suffix in [".json", ".jsonl"]:
        df = pd.read_json(corpus_path, lines=True)
    else:
        logger.info(
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
        top_docs, _ = doc_selector.get_top_docs(
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
