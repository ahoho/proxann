import logging
import pathlib
import random
from itertools import product
from typing import List, Optional, Union

import gensim.downloader as api
import numpy as np

from src.utils.utils import init_logger


class TopicSelector(object):
    """
    Class to select topics from different topic models.
    """

    def __init__(
        self,
        wmd_model: str = 'word2vec-google-news-300',
        logger: Optional[logging.Logger] = None,
        config_path: pathlib.Path = pathlib.Path("config/config.yaml")
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
        self._logger = logger if logger else init_logger(config_path, __name__)
        self._logger.info("Loading Word2Vec model...")
        self._wmd_model = api.load(wmd_model)
        self._logger.info("Word2Vec model loaded.")

        return

    def _get_wmd(self, from_: Union[str, List[str]], to_: Union[str, List[str]], n_words=10) -> float:
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

        return self._wmd_model.wmdistance(from_, to_)

    def _get_wmd_mat(self, models: list) -> np.ndarray:
        """Calculate inter-topic distance based topic words using Word Mover's Distance.

        Parameters
        ----------
        models : list
            A list containing two sublists the models. Each sublits is a list of topics, each topic represented as a list of words.

        Returns
        -------
        np.ndarray
            A matrix of Word Mover's Distance between topics from two models.
        """

        if len(models) != 2:
            raise ValueError(
                "models must contain exactly two sublists/arrays.")

        num_topics_first_model = len(models[0])
        num_topics_second_model = len(models[1])
        wmd_sims = np.zeros((num_topics_first_model, num_topics_second_model))

        for k_idx, k in enumerate(models[0]):
            for k__idx, k_ in enumerate(models[1]):
                wmd_sims[k_idx, k__idx] = self._get_wmd(k, k_)

        return wmd_sims

    def iterative_matching(self, models, N, remove_topic_ids=None, seed=2357_11):
        """
        Performs an iterative pairing process between the topics of multiple models.

        Parameters
        ----------
        models : list
            A list containing two sublists the models. Each sublits is a list of topics, each topic represented as a list of words.
        N : int
            Number of matches to find.

        Returns
        -------
        list of list of tuple
            List of lists with the N matches found. Each match is a list of tuples, where each tuple contains the model index and the topic index.
        """

        if remove_topic_ids is not None:
            modified_models = []
            id_mappings = []  # To store mappings for each model
            for i_model, model in enumerate(models):
                # Create a mapping for this model
                mapping = {}
                new_model = []
                new_topic_id = 0
                for i, topic in enumerate(model):
                    if len(remove_topic_ids) > 1 and i not in remove_topic_ids[i_model]:
                        new_model.append(topic)
                        # Map new topic ID to the original
                        mapping[new_topic_id] = i
                        new_topic_id += 1
                modified_models.append(new_model)
                id_mappings.append(mapping)  # Store the mapping for this model
            models = modified_models
        random.seed(seed)
        dists = {}
        for modelA, modelB in product(range(len(models)), range(len(models))):
            dists[(modelA, modelB)] = self._get_wmd_mat(
                [models[modelA], models[modelB]])

        matches = []  # Matches with filtered topic IDs

        assert (all(N <= len(m) for m in models))
        while len(matches) < min(len(m) for m in models):
            for seed_model in range(len(models)):
                # Calculate the mean distance to all other models
                min_dists, min_dists_indices = [], []
                for other_model in range(len(models)):
                    if seed_model == other_model:
                        min_dists_indices.append((seed_model, None))
                        continue
                    distsAB = dists[(seed_model, other_model)]
                    # Get the minimum distance for each topic in the seed model to the other model
                    min_dists.append(distsAB.min(1))
                    min_dists_indices.append((other_model, distsAB.argmin(1)))
                mean_min_dists = np.mean(min_dists, axis=0)
                seed_model_topic = np.argmin(mean_min_dists)
                seed_model_matches = [
                    (model_idx, indices[seed_model_topic]) if model_idx != seed_model else (
                        model_idx, seed_model_topic)
                    for model_idx, indices in min_dists_indices
                ]
                matches.append(seed_model_matches)

                # Remove the matched topics from the distance matrix
                for modelA, modelA_topic in seed_model_matches:
                    for modelB in range(len(models)):
                        if modelA != modelB:
                            dists[(modelA, modelB)][modelA_topic, :] = np.inf
                            dists[(modelB, modelA)][:, modelA_topic] = np.inf

        sampled_matches = random.sample(matches, N)

        # Map the sampled matches to their original topic IDs (sampled_matches are just positions in the betas matrix)
        sampled_matches_original = [
            [
                (model_idx, id_mappings[model_idx][topic_id]
                 if topic_id is not None else None)
                for model_idx, topic_id in match
            ]
            for match in sampled_matches
        ]

        # Output the sampled matches in both forms
        print("Sampled Matches (Position IDs):", sampled_matches)
        print("Sampled Matches (Original Topic IDs):", sampled_matches_original)

        return sampled_matches_original
