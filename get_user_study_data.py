"""Extracts information from a set of topic models provided in a configuration file and saves the output in a JSON file.

From each model, as many topics as specified in the 'n_matches' parameter of the configuration file are kept. The topics are selected using the iterative matching algorithm implemented in the TopicSelector class. 

For each of the selected topics, the top words, exemplar documents, evaluation documents, and a distractor document are extracted.

Then, using the path of the model as key, the output of all models is grouped in a dictionary and save it in a JSON file.
"""

import argparse
from src.proxann.proxann import ProxAnn

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config_path",
        help="Path to the configuration file.",
        type=str,
        required=False,
        default="config/config.yaml"
    )
    argparser.add_argument(
        "--user_study_config",
        help="Path to the user study configuration file.",
        required=False,
        default="config/config_pilot.conf")

    args = argparser.parse_args()
    
    proxann = ProxAnn(config_path=args.config_path)

    proxann.generate_user_provided_json(path_user_study_config_file=args.user_study_config)


if __name__ == "__main__":
    main()
