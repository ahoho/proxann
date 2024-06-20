import argparse
import logging
import pathlib
from typing import List, Optional, Union
import gensim.downloader as api
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from src.utils.utils import init_logger


class TopicSelector(object):
    """
    Class to select topics from different topic models.
    """

    def __init__(
        self,
        wmd_model: str = 'word2vec-google-news-300',
        logger: Optional[logging.Logger] = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ) -> None:
        """
        Initialize the TopicSelector class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger object to log activity.
        path_logs : pathlib.Path, optional
            Path for saving logs.
        """
        self._logger = logger if logger else init_logger(__name__, path_logs)
        self._wmd_model = api.load(wmd_model)      

        return

    def _bhatta(self, vector1, vector2):
        """
        Calculates the Bhattacharyya coefficient between two vectors.

        Parameters
        ----------
        vector1 : numpy.ndarray
            First vector.
        vector2 : numpy.ndarray
            Second vector.

        Returns
        -------
        float
            Bhattacharyya coefficient.
        """
        return np.sum(np.sqrt(vector1 * vector2))

    def _largest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        idx0, idx1 = np.unravel_index(indices, ary.shape)
        idx0 = idx0.tolist()
        idx1 = idx1.tolist()
        selected_idx = []
        for id0, id1 in zip(idx0, idx1):
            if id0 < id1:
                selected_idx.append((id0, id1, ary[id0, id1]))
        return selected_idx

    def _jensen_sim(self, mat1, mat2):
        """Computes inter-topic distance based on word distributions using Jensen Shannon distance.
        """

        js_mat = np.zeros((mat1.shape[0], mat2.shape[0]))
        for k in range(mat1.shape[0]):
            for kk in range(mat2.shape[0]):
                js_mat[k, kk] = jensenshannon(
                    mat1[k, :], mat2[kk, :])
        JSsim = 1 - js_mat

        return JSsim
    
    def _get_wmd(
        self,
        from_: Union[str, List[str]],
        to_: Union[str, List[str]],
        n_words=10
    ) -> float:
        """
        Calculate the Word Mover's Distance between two sentences.

        Parameters
        ----------
        from_ : Union[str, List[str]]
            The source sentence.
        to_ : Union[str, List[str]]
            The target sentence.
        n_words : int
            The number of words to consider in the sentences to calculate the WMD.
        """
        if isinstance(from_, str):
            from_ = from_.split()

        if isinstance(to_, str):
            to_ = to_.split()

        if n_words < len(from_):
            from_ = from_[:n_words]
        if n_words < len(to_):
            to_ = to_[:n_words]

        #self._logger.info(f"-- -- Calculating WMD from: {from_} to {to_}")

        return self._wmd_model.wmdistance(from_, to_)
    

    def find_most_similar_pairs(self, mat1, mat2):
        """
        Finds the optimal pairings between topics using the Hungarian algorithm.

        Parameters
        ----------
        JSsim : numpy.ndarray
            Matrix of Jensen-Shannon similarities between topics from two models.

        Returns
        -------
        list of tuple
            List of tuples with the optimal pairings
        """

        JSsim = self._jensen_sim(mat1, mat2)

        # linear_sum_assignment finds the minimum cost, so we need to convert the similarity matrix to a cost matrix.
        cost_matrix = -JSsim
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        results = [(i, j, JSsim[i, j]) for i, j in zip(row_ind, col_ind)]
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        pairings = [(i, j) for i, j, _ in sorted_results]

        return pairings

    def find_most_dissimilar_pairs(self, mat1, mat2):
        """
        Finds the most dissimilar pairs using the Hungarian algorithm.


        Parameters
        ----------
        mat1 : numpy.ndarray
            Matrix of topics from the first model.
        mat2 : numpy.ndarray
            Matrix of topics from the second model.

        Returns
        -------
        list of tuple
            List of tuples with the most dissimilar pairs
        """

        JSsim = self._jensen_sim(mat1, mat2)

        # Hungarian algorithm on dissimilarity matrix
        dissimilarity_matrix = 1 - JSsim
        row_ind, col_ind = linear_sum_assignment(dissimilarity_matrix)

        pairs = list(zip(row_ind, col_ind))
        pairs = [(int(i), int(j)) for i, j in pairs]

        return pairs

    def _get_models_combinations(self, models):
        """
        Generates pairs of elements in a circular manner from a given list.

        This function takes a list of 'models' and returns a list of tuples,
        where each tuple represents a pair of adjacent models in a circular
        manner. The last element pairs with the first element to ensure the 
        circular behavior.

        Parameters
        ----------
        models : list
            List of models to generate pairs from.

        Returns
        -------
        list of tuple
            List of tuples with pairs of models.

        Example:
        self._get_models_combinations(['modelA', 'modelB', 'modelC', 'modelD'])
        [('modelA', 'modelB'), ('modelB', 'modelC'), ('modelC', 'modelD'), ('modelD', 'modelA')]
        """
        combs = []
        for i in range(len(models)):
            combs.append((models[i], models[(i+1) % len(models)]))
        return combs

    def iterative_matching(self, models, N):
        """
        Performs an iterative pairing process between the topics of multiple models.

        Parameters
        ----------
        models : list of numpy.ndarray
            List with the betas matrices from different models.
        N : int
            Number of matches to find.

        Returns
        -------
        list of list of tuple
            List of lists with the N matches found. Each match is a list of tuples, where each tuple contains the model index and the topic index.
        """

        # Get topic pairs for each pair of models
        topic_pairs = {}
        combs = self._get_models_combinations(list(range(len(models))))

        for modelA, modelB in combs:
            self._logger.info(
                f"Calculating pairing between {modelA} and {modelB}..")
            if (modelA, modelB) not in topic_pairs:
                topic_pairs[(modelA, modelB)] = self.find_most_similar_pairs(
                    models[modelA], models[modelB])
                # Cache the inverted pairs for reverse lookups
                topic_pairs[(modelB, modelA)] = [(pair[1], pair[0])
                                                 for pair in topic_pairs[(modelA, modelB)]]
            else:
                self._logger.info(
                    f"Pairing between {modelA} and {modelB} already calculated.")

        num_models = len(models)
        matches = []

        while len(matches) < N:
            this_model_matches = []
            for start_model in range(num_models):
                if len(matches) >= N:
                    break

                this_model_matches = []
                used_topics = set()

                current_model = start_model
                while len(this_model_matches) < num_models:
                    next_model = (current_model + 1) % num_models

                    if not topic_pairs[(current_model, next_model)]:
                        break  # Skip if no pairs are left

                    match = next((pair for pair in topic_pairs[(
                        current_model, next_model)] if pair[0] not in used_topics), None)

                    if match:
                        topic_pairs[(current_model, next_model)].remove(match)
                        this_model_matches.append((current_model, match[0]))
                        used_topics.add(match[0])
                        current_model = next_model
                    else:
                        break

                if len(this_model_matches) == num_models:
                    matches.append(this_model_matches)
                    break

        return matches
    
    def find_closest_by_wmd(self, models: list, keep_from_first: list = [0, 1, 2]) -> list:
        """Find the closest topics between two models using Word Mover's Distance.
        
        Parameters
        ----------
        models : list
            A list containing two sublists/arrays representing the models.
        keep_from_first : list, optional
            Indices of topics from the first model to keep, by default [0, 1, 2]
        
        Returns
        -------
        list
            A list of tuples, each containing pairs of (model_index, topic_index) indicating the closest topics.
        """
        
        if len(models) != 2:
            raise ValueError("models must contain exactly two sublists/arrays.")
        
        num_topics_first_model = len(models[0])
        num_topics_second_model = len(models[1])
        
        wmd_sims = np.zeros((num_topics_first_model, num_topics_second_model))
        
        for k_idx, k in enumerate(models[0]):
            for k__idx, k_ in enumerate(models[1]):
                wmd_sims[k_idx, k__idx] = self._get_wmd(k, k_)
        
        tuple_results = []
        selected_k_indices = set()  # To keep track of selected indices from the second model
        
        for k in keep_from_first:
            closest_k = None
            closest_distance = float('inf')
            for k__idx in range(num_topics_second_model):
                if k__idx not in selected_k_indices and wmd_sims[k, k__idx] < closest_distance:
                    closest_k = k__idx
                    closest_distance = wmd_sims[k, k__idx]
                    
            if closest_k is not None:
                selected_k_indices.add(closest_k)
                tuple_results.append([(0, k), (1, closest_k)])
        
        return tuple_results

                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--betas_paths", help="Paths of the models'betas files, separated by commas", required=True)
    parser.add_argument(
        "--N", type=int, help="Number of matches (topics to evaluate)", default=2)

    betas_mats = []
    args = parser.parse_args()
    for betas_path in args.betas_paths.split(","):
        betas = np.load(betas_path)
        if betas.shape[0] > betas.shape[1]:
            betas = betas.T
        betas_mats.append(betas)

    # get K for each model
    """
    Ks = [betas.shape[0] for betas in betas_mats]
    # Find if there is a model with a different K, and if so, get that model index
    diff_K = [K for K in Ks if K != Ks[0]]
    for idx_diff in diff_K:
        print(f"Modelo {idx_diff} tiene un K diferente")
        # Compensate the K difference with zero rows
        betas_mats[idx_diff] = np.vstack((betas_mats[idx_diff], np.zeros((Ks[0] - Ks[idx_diff], betas_mats[idx_diff].shape[1]))))
    """

    selector = TopicSelector()
    matches = selector.iterative_matching(betas_mats, args.N)

    print("Matches:", matches)


if __name__ == '__main__':
    main()
