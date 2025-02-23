"""
Main script for evaluating a topic using the Proxann model. 

Usage:
------
$ python proxann_eval.py --config_path config/config.yaml --user_study_config config/config_pilot.conf

Arguments:
----------
- `config_path` (str): Path to the configuration file.
- `user_study_config` (str): Path to the user study configuration file, with the following format:

    1. If the model has been trained using Proxann, only the paths to the trained model and the dataset are required. Set "trained_with_thetas_eval=True".
    
    Example:
    --------
    model_path = "data/models/mallet"
    corpus_path = "data/training_data/bills/train.metadata.embeddings.jsonl.all-MiniLM-L6-v2.parquet"
    trained_with_thetas_eval = True

    2. If the model was trained separately, you must specify additional paths:
    - Thetas file (as a sparse matrix)
    - Betas file
    - Vocabulary file (JSON)``
    - Corpus file
    - Set `trained_with_thetas_eval=False`

    Example:
    --------
    mallet_config = {
    "thetas_path": "data/models/mallet/doctopics.npz.npy",
    "betas_path": "data/models/mallet/beta.npy",
    "vocab_path": "data/models/mallet/vocab.json",
    "corpus_path": "data/train.metadata.jsonl",
    "trained_with_thetas_eval": False
    }
"""

import argparse

from src.proxann.proxann import ProxAnn
from src.utils.utils import init_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str,  required=False, default="config/config.yaml",
        help="Path to the configuration file.")
    parser.add_argument(
        "--user_study_config",
        help="Path to the user study configuration file.",
        required=False,
        default="config/config_pilot.conf")
    
    return parser.parse_args()


def main():
    args = parse_args()

    logger = init_logger(args.config_path, f"RunProxann-eval")
    logger.info(f"Running Proxann in metric mode")
    
    # Init proxann object
    proxann = ProxAnn(logger, args.config_path)

    # Generate user provided JSON file
    status, tm_model_data_path = proxann.generate_user_provided_json(path_user_study_config_file=args.user_study_config)
    
    if status == 0:
        logger.info("User provided JSON file generated successfully.")
    else:
        logger.error("Error generating user provided JSON file.")
        return 1
    
    proxann.run_metric(
        tm_model_data_path.as_posix(),
        llm_models=["qwen:32b"]
    )
    

if __name__ == "__main__":
    main()
