from collections import defaultdict
import configparser
import itertools
import json
import logging
import math
import pathlib
import random
import re
import sys

from typing import List, Optional, Union

import numpy as np
import pandas as pd  # type: ignore
from scipy import sparse
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score  # type: ignore

from src.proxann.prompter import Prompter
from src.proxann.utils import (
    bradley_terry_model,
    extend_to_full_sentence,
    extract_info_binary_q2,
    extract_info_mean_q2,
    extract_info_mean_q3,
    extract_info_q1_q3,
    extract_logprobs,
    load_config_pilot,
    load_template
)
from src.user_study_data_collector.jsonify.topic_json_formatter import TopicJsonFormatter
from src.user_study_data_collector.topics_docs_selection.topic_selector import TopicSelector
from src.utils.utils import init_logger, load_vocab_from_txt, load_yaml_config_file, log_or_print

class ProxAnn(object):
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent / "config/config.yaml"
    ) -> None:
        """Initialize ProxAnn object.
        
        Parameters
        ----------
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        config_path : pathlib.Path, optional
            Path to the configuration file, by default pathlib.Path(__file__).parent.parent.parent / "config/config.yaml"    
        """
        self._logger = logger if logger else init_logger(config_path, __name__)
        
        self._logger.info("ProxAnn logger object initialized.")
        
        # load config from user_study
        self._config = load_yaml_config_file(config_path, "user_study", logger)
        
        self._logger.info("ProxAnn object initialized.")
        
        
        return
    
    def generate_user_provided_json(
        self,
        path_user_study_config_file: str,
        output_path: str = None,
        user_provided_tpcs: Optional[List[int]] = None,
    ) -> int:
        """Generate a JSON file with user-provided model data.
        
        We assume the user gives the <path_user_study_config_file> already formatted or that the backend generates it accordingly based on the files uploaded by the user.
        
        Parameters
        ----------
        path_user_study_config_file : str
            Path to the user study configuration file.
        output_path : str
            Path to save the JSON file.
        user_provided_tpcs : Optional[List[int]], optional
            List of user-provided topics, by default None. If not provided, the function will use the topics from the configuration file.
        """
        self._logger.info(
            "Generating JSON file with user-provided model data starts...")
        
        user_config = configparser.ConfigParser()
        user_config.read(path_user_study_config_file)
        self._logger.info(f"User study configuration read successfully")
        
        ############################
        # Topic selection          #
        ############################
        N = int(user_config['all']['n_matches'])
        top_words_display = int(user_config['all']['top_words_display'])
        
        # if the user provides topics, we do not need to select them, but we need to return them in the same format the topic selector uses
        # if the user provides topics, then we only allow for one model to be evaluated, so we need to check that if user_provided_tpcs is not None, then the user config only has one model
        if user_provided_tpcs is not None:
            # check if the user config has more than one model
            if len(user_config.sections()) > 2:
                error_msg = f"User provided topics, but more than one model is configured. Exiting..."
                log_or_print(error_msg, self._logger)
                return 1, error_msg
            
            selected_topics = [[(0, user_provided_tpcs[i]) for i in range(len(user_provided_tpcs))]]
            log_or_print(f"Selected topics set from user input: {selected_topics}", self._logger)
        else:

            remove_topic_ids = []
            all_topic_keys = []
            for model in user_config.sections():
                # Skip all section (configuration for all models)
                if model == 'all':
                    continue
                model_config = user_config[model]
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
                        vocab_w2id = load_vocab_from_txt(vocab_path)
                    else:
                        log_or_print(
                            f"File does not have the required extension for loading the vocabulary. Exiting...", self._logger)
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
        
            log_or_print(f"Selected topics with TopicSelector: {selected_topics}", self._logger)

        ############################
        # JSON formatter           #
        ############################
        method = user_config['all']['method']
        ntop = int(user_config['all']['ntop'])
        text_column = user_config['all']['text_column']
        text_column_disp = user_config['all']['text_column_disp']
        thr = user_config['all']['thr']
        thr = float(thr.split(",")[0]), float(thr.split(",")[1])

        formatter = TopicJsonFormatter()

        idx_model = 0
        combined_out = {}
        for model in user_config.sections():
            model_config = user_config[model]
            if model != 'all':  # Skip all section (configuration for all models)
                model_config = user_config[model]
                log_or_print(f"Obtaining output for model {model}", self._logger)
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
                        log_or_print(
                            f"Error occurred when loading info from model {model_path.as_posix(): e}", self._logger)
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
                        vocab_w2id = load_vocab_from_txt(vocab_path)
                    else:
                        log_or_print(
                            f"File does not have the required extension for loading the vocabulary. Exiting...", self._logger)
                        sys.exit()
                
                # check the number of topics selected for the models is less or equal to the number of topics in the model, and topics are valid IDs (all ids go from 0 to number of topics - 1)
                this_model_tpcs = [
                    el[1] for el in selected_topics if el[0] == idx_model]
                
                if len(this_model_tpcs) > thetas.shape[0]:
                    error_msg = f"Number of topics selected for model {model} is greater than the number of topics in the model. Exiting..."
                    log_or_print(error_msg, self._logger)
                    return 1, error_msg
                
                for el in this_model_tpcs:
                    if el < 0 or el >= thetas.shape[1]:
                        error_msg = f"Topic ID {el} is not valid for model {model}. Exiting..."
                        log_or_print(error_msg, self._logger)
                        return 1, error_msg
                    
                #  Get keys
                vocab_id2w = dict(zip(vocab_w2id.values(), vocab_w2id.keys()))
                keys = [
                    [vocab_id2w[idx]
                        for idx in row.argsort()[::-1][:top_words_display]]
                    for row in betas
                ]
                # print id and top words
                log_or_print(f"Top words for model {model}", self._logger)
                for i, key in enumerate(keys):
                    log_or_print(f"* Topic {i}: {key[:10]}", self._logger)

                #  Get corpus
                corpus_path = pathlib.Path(model_config['corpus_path'])
                if corpus_path.suffix == ".parquet":
                    df = pd.read_parquet(corpus_path)
                elif corpus_path.suffix in [".json", ".jsonl"]:
                    df = pd.read_json(corpus_path, lines=True)
                else:
                    log_or_print(
                        f"Unrecognized file extension for data path: {corpus_path.suffix}. Exiting...", self._logger)
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

                log_or_print(
                    f"Topics to keep for model {model}: {this_model_tpc_to_keep}", self._logger)

                # Filter JSON output
                filtered_out = {int(key): out[key]
                                for key in this_model_tpc_to_keep if key in out}

                if not model_path:
                    model_path = pathlib.Path(thetas_path).parent.as_posix()
                log_or_print(
                    f"Output for model {model} has {len(filtered_out)} topics", self._logger)
                log_or_print(
                    f"Saving output for model {model} with key {model_path}", self._logger)

                combined_out.update({
                    model_path: filtered_out
                })

                idx_model += 1

        if output_path is None:
            path_json_save = user_config['all']['path_json_save']
            output_path = pathlib.Path(path_json_save) / (pathlib.Path(path_user_study_config_file).stem + ".json")
            
        # Write JSON output to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(combined_out, file, indent=4)    
        
        if output_path.exists():
            return 0, output_path
        else:
            return 1, "Error writing JSON file"
        
    def get_prompt_template(
        self,
        llm_model_type: str,
        text_for_prompt: dict,
        question_type: str,
        category: str = None,
        nr_few_shot: int = 2,
        doing_both_ways: bool = False,
        generate_description: bool = False,
        path_examples: str = "src/proxann/prompts/few_shot_examples.json",
        base_prompt_path: str = "src/proxann/prompts/"
    ) -> Union[str, List[str]]:

        # Read JSON with few-shot examples
        with open(path_examples, 'r') as file:
            few_shot_examples = json.load(file)

        def load_template_path(q_type: str) -> str:
            """Helper function to construct the template path based on question type."""
            return f"{base_prompt_path}{q_type}/instructions_question_prompt.txt"
        
        def add_model_key_to_template(template_name: str) -> str:
            """Helper function to add the model key to the template."""
            
            if "q3" in template_name:
                # TODO: Update this when q3 is optimized for Q3
                return f"{template_name}_llama" if "llama" in llm_model_type else template_name
            else:
                if "llama" in llm_model_type:
                    return f"{template_name}_llama"
                elif "qwen" in llm_model_type:
                    return f"{template_name}_qwen"
                elif "gpt" in llm_model_type:
                    return template_name
            
                self._logger.info("No specific template for the model type. Using the default template.")
            return template_name

        def handle_q1(
            text: dict, 
            few_shot: dict, 
            topk: int = 10,
            num_words: int = 100,
            nr_few_shot=1,
            generate_description: bool = False
        ) -> str:
            """Function to handle Q1."""
            
            # Few-shot examples
            examples_q1 = few_shot["q1"][:nr_few_shot]

            if generate_description:
                examples = [
                    "KEYWORDS: {}\nDOCUMENTS: {}\nCATEGORY: {}\nDESCRIPTION:{}".format(
                        ex['example']['keywords'],
                        ''.join(
                            f"\n- {doc}" for doc in ex['example']['documents']),
                        ex['example']['response']['label'],
                        ex['example']['response']['description']
                    )
                    for ex in examples_q1
                ]
            else:
                examples = [
                    "KEYWORDS: {}\nDOCUMENTS: {}\nCATEGORY: {}".format(
                        ex['example']['keywords'],
                        ''.join(
                            f"\n- {doc}" for doc in ex['example']['documents']),
                        ex['example']['response']['label']
                    )
                    for ex in examples_q1
                ]

            # Actual question to the LLM (keys and docs)
            docs = "".join(
                f"\n- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])

            keys = " ".join(text["topic_words"][:topk])

            template_path = load_template_path(
                "q1_with_desc") if generate_description else load_template_path("q1")
            return load_template(template_path).format("\n".join(examples), keys, docs)

        def handle_q2(
            text: dict,
            cat: str,
            num_words: int = 100,
            template_name: str = "q2_dspy"
        ) -> List[str]:
            """Function to handle Q2."""

            if template_name == "q2_dspy":
                template_name = add_model_key_to_template(template_name)
            template_path = load_template_path(template_name)
            return [load_template(template_path).format(category=cat, document=extend_to_full_sentence(doc['text'], num_words)) for doc in text["eval_docs"]]

        def handle_q3(
            text: dict,
            cat: str,
            num_words: int = 100,
            doing_both_ways: bool = True,
            template_name: str = "q3_dspy",
            return_raw: bool = False
        ) -> List[str]:
            """Function to handle Q3."""

            # Set seed for consistent randomness
            random.seed(1234)

            # Document pairs generation
            doc_pairs_one, doc_pairs_two = [], []
            pair_ids_one, pair_ids_two = [], []

            eval_docs = sorted(text["eval_docs"], key=lambda x: x["doc_id"])
            # shuffle the eval docs so the "most related" ones are not shown first
            eval_docs = random.sample(eval_docs, len(eval_docs))
            for d1, d2 in itertools.combinations(eval_docs, 2):
                docs_list = [[d1, d2], [d2, d1]] if doing_both_ways else [
                    random.sample([d1, d2], 2)]

                for i, docs in enumerate(docs_list):
                    doc_a = extend_to_full_sentence(docs[0]['text'], num_words)
                    doc_b = extend_to_full_sentence(docs[1]['text'], num_words)

                    if doing_both_ways:
                        if i % 2 == 0:
                            doc_pairs_one.append([doc_a, doc_b])
                            pair_ids_one.append(
                                {"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                        else:
                            doc_pairs_two.append([doc_a, doc_b])
                            pair_ids_two.append(
                                {"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                    else:
                        doc_pairs_one.append([doc_a, doc_b])
                        pair_ids_one.append(
                            {"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})

            # Template loading and formatting
            if template_name == "q3_dspy":
                template_name = add_model_key_to_template(template_name)
            template_path = load_template_path(template_name)
            template = load_template(template_path)
            
            if "with_eval" in template_name:
                template = template.format(
                    keywords = " ".join(text["topic_words"][:7]),
                    exemplar_docs = "\n".join(
                        f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"][:4]),
                    category= "{category}",
                    doc_a="{doc_a}",
                    doc_b="{doc_b}"
                )

            if doing_both_ways:
                return (
                    [template.format(category=cat, doc_a=pair[0], doc_b=pair[1]) for pair in doc_pairs_one],
                    pair_ids_one,
                    [template.format(category=cat, doc_a=pair[0], doc_b=pair[1]) for pair in doc_pairs_two],
                    pair_ids_two
                )
            else:
                if not return_raw:
                    return ([template.format(category=cat, doc_a=pair[0], doc_b=pair[1]) for pair in doc_pairs_one], pair_ids_one)
                else:
                    return (cat, [(pair[0], pair[1]) for pair in doc_pairs_one], pair_ids_one)

        question_handlers = {
            "q1": lambda text: handle_q1(text, few_shot_examples, nr_few_shot=nr_few_shot, generate_description=generate_description),
            "q2_dspy": lambda text: handle_q2(text, category, template_name = "q2_dspy"),
            "q3_dspy": lambda text: handle_q3(text, category, doing_both_ways=doing_both_ways),
            "q2_mean": lambda text: handle_q2(text, category, template_name= "q2_mean"),
            "q3_mean": lambda text: handle_q3(text, category, doing_both_ways=doing_both_ways, template_name= "q3_mean"),
            "q3_mean_with_eval": lambda text: handle_q3(text, category, doing_both_ways=doing_both_ways, template_name= "q3_mean_with_eval"),
        }

        if question_type not in question_handlers:
            raise ValueError("Invalid question type")

        return question_handlers[question_type](text_for_prompt)
    
    # =========================================================================
    # Logic for Q1 / Q2 / Q3
    # =========================================================================
    def do_q1(
        self,
        prompter: Prompter,
        cluster_data: dict,
        users_cats: list,
        categories: list,
        temperature: float = None,
        dft_system_prompt: str = "src/proxann/prompts/q1/simplified_system_prompt.txt",
        logger: logging.Logger = None
    ) -> None:
        """Execute Q1.

        Parameters
        ----------
        prompter : Prompter
            Prompter object.
        cluster_data : Information for a topic as given by the data loaded from args.tm_model_data_path. 
        users_cats : List of user-generated categories during the user study.
        categories : List to store the LLM-generated categories.
        temperature : float, optional
            Custom temperature for the LLM, by default None. If not given, the default temperature from the prompter object is used.
        dft_system_prompt : str, optional
            Default system prompt for Q1, by default "src/proxann/prompts/q1/simplified_system_prompt.txt".
        logging : logging.Logger, optional
            Logger object, by default None
        """
        log_or_print("Executing Q1...", logger)

        if temperature is not None:
            self._logger.info(f"Using temperature {temperature} for Q1.")

        question = self.get_prompt_template(prompter.model_type,cluster_data, "q1")
        category, _ = prompter.prompt(
            dft_system_prompt, question, use_context=False, temperature=temperature)

        categories.append(category)
        if users_cats != []:
            log_or_print(f"\033[94mUser categories: {users_cats}\033[0m", logger)
        log_or_print(f"\033[94mModel category: {category}\033[0m", logger)

        return

    def do_q2(
        self,
        prompter: Prompter,
        prompt_mode: str,
        cluster_data: dict,
        fit_data: list,
        category: str,
        user_cats: list = None,
        temperature: float = None,
        dft_system_prompt: str = None,
        use_context: bool = False,
        logger: logging.Logger = None
    ) -> list:
        """
        Execute Q2.

        Parameters
        ----------
        prompter : Prompter
            Prompter object.
        prompt_mode : str
            Prompting mode for Q2.
        cluster_data : dict
            Information for a topic as given by the data loaded from args.tm_model_data_path.
        fit_data : list
            List to store the fit scores.
        category : str
            LLM-generated category.
        user_cats : list, optional
            List of user-generated categories during the user study, by default None.
        temperature : float, optional
            Custom temperature for the LLM, by default None. If not given, the default temperature from the prompter object is used.
        dft_system_prompt : str, optional
            Default system prompt for Q2.
        use_context : bool, optional
            Whether to use the context for the prompt, by default False.
        logger : logging.Logger, optional
            Logger for logging information.

        Returns
        -------
        list
            List of user categories.
        """

        if temperature is not None:
            self._logger.info(f"Using temperature {temperature} for Q2.")
                    
        if prompt_mode not in ["q1_then_q2_dspy", "q1_then_q2_mean"]:
            log_or_print(f"Not a valid prompt mode: {prompt_mode}", logger)
            return

        prompt_key = "q2_mean" if prompt_mode == "q1_then_q2_mean" else "q2_dspy"
        log_or_print(f"Using prompt key: {prompt_key}", logger)

        if user_cats:
            labels = user_cats * len(self.get_prompt_template(prompter.model_type, cluster_data, prompt_key, category))

            for cat in user_cats:
                questions = self.get_prompt_template(prompter.model_type,cluster_data, prompt_key, cat)
                
                for question in questions:
                    response_q2, logprobs_q2 = prompter.prompt(dft_system_prompt, question, use_context=use_context, temperature=temperature)
                    if prompt_key == "q2_dspy":
                        score = extract_info_binary_q2(response_q2)
                    elif prompt_key == "q2_mean":
                        score = extract_info_mean_q2(logprobs_q2)
                    else:
                        raise ValueError("Invalid prompt key for Q2.")
                    log_or_print(f"\033[92mFit: {score}\033[0m", logger)
                    fit_data.append(score)
        else:
            questions = self.get_prompt_template(prompter.model_type,cluster_data, prompt_key, category)
            labels = [category] * len(questions)

            for question in questions:
                response_q2, logprobs_q2 = prompter.prompt(dft_system_prompt, question, use_context=use_context, temperature=temperature)
                if prompt_key == "q2_dspy":
                        score = extract_info_binary_q2(response_q2)
                elif prompt_key == "q2_mean":
                    score = extract_info_mean_q2(logprobs_q2)
                else:
                    raise ValueError("Invalid prompt key for Q2.")
                log_or_print(f"\033[92mFit: {score}\033[0m", logger)
                fit_data.append(score)

        return labels

    def do_q3(
        self,
        prompter: Prompter,
        prompt_mode: str,
        cluster_data: dict,
        rank_data: list,
        info_to_bradley_terry: dict,
        users_rank: list,
        category: str,
        temperature: float = None,
        doing_both_ways: bool = True,
        dft_system_prompt: str = "src/proxann/prompts/q3_dspy/simplified_system_prompt.txt",
        use_context: bool = False,
        logger: logging.Logger = None
    ) -> None:
        """
        Execute Q3.

        Parameters
        ----------
        prompter : Prompter
            Prompter object.
        prompt_mode : str
            Prompting mode for Q3.
        cluster_data : dict
            Information for a topic as given by the data loaded from args.tm_model_data_path.
        rank_data : list
            List to store the rank data.
        users_rank : list
            List of user ranks.
        category : str
            LLM-generated category.
        temperature : float, optional
            Custom temperature for the LLM, by default None. If not given, the default temperature from the prompter object is used.
        doing_both_ways : bool, optional
            Whether to run Q3 twice: once with A as the first document, then reversed, by default False.
        dft_system_prompt : str, optional
            Default system prompt for Q3, by default "src/proxann/prompts/q3/simplified_system_prompt.txt".
        use_context : bool, optional
            Whether to use the context for the prompt, by default False.
        logger : logging.Logger, optional
            Logger object, by default None.
        """
                
        if temperature is not None:
            self._logger.info(f"Using temperature {temperature} for Q3.")
        
        if prompt_mode not in ["q1_then_q3_dspy", "q1_then_q3_mean"]:
            log_or_print(f"Not a valid prompt mode: {prompt_mode}", logger)
            return
        
        prompt_key = "q3_mean" if prompt_mode == "q1_then_q3_mean" else "q3_dspy"
        
        keep_only_most_top = False
        log_or_print(f"-- Using prompt key: {prompt_key}", logger)
        q3_out = self.get_prompt_template(prompter.model_type, cluster_data, prompt_key, category=category, doing_both_ways=doing_both_ways)

        if isinstance(q3_out, tuple) and len(q3_out) > 2:  # Both ways
            questions_one, pair_ids_one, questions_two, pair_ids_two = q3_out
            ways = [[questions_one, pair_ids_one], [questions_two, pair_ids_two]]
        else:  # Single way
            questions, pair_ids = q3_out
            ways = [[questions, pair_ids]]

        labels_one, orders_one, rationales_one, logprobs_one, all_info_logprobs_one = [], [], [], [], []
        labels_two, orders_two, rationales_two, logprobs_two, all_info_logprobs_two = [], [], [], [], []

        for way_id, (questions, pair_ids) in enumerate(ways):
            log_or_print(
                f"-- Executing Q3 ({'both ways' if len(ways) > 1 else 'one way'})...", logger)
            for question in questions:
                pairwise, pairwise_logprobs = prompter.prompt(
                    dft_system_prompt, question, use_context=use_context, temperature=temperature
                )

                try:
                    label, order, rationale = extract_info_q1_q3(
                        pairwise, get_label=(prompt_mode == "q1_and_q3"))
                    if len(ways) > 1 and way_id == 0:
                        labels_one.append(label)
                        orders_one.append(order)
                        rationales_one.append(rationale)
                        logprobs_one.append(extract_logprobs(
                            pairwise_logprobs, prompter.backend, logger))
                        all_info_logprobs_one.append(pairwise_logprobs)
                        log_or_print(f"\033[92mOrder -->: {order}\033[0m", logger)
                    elif len(ways) > 1 and way_id == 1:
                        labels_two.append(label)
                        orders_two.append(order)
                        rationales_two.append(rationale)
                        logprobs_two.append(extract_logprobs(
                            pairwise_logprobs, prompter.backend, logger))
                        all_info_logprobs_two.append(pairwise_logprobs)
                        log_or_print(f"\033[94mOrder <--: {order}\033[0m", logger)
                    else:
                        labels_one.append(label)
                        orders_one.append(order)
                        rationales_one.append(rationale)
                        logprobs_one.append(extract_logprobs(
                            pairwise_logprobs, prompter.backend, logger))
                        all_info_logprobs_one.append(pairwise_logprobs)
                        log_or_print(f"\033[92mOrder: {order}\033[0m", logger)
                except Exception as e:
                    log_or_print(
                        f"-- Error extracting info from prompt: {e}", "error", logger)   
        # Combine results for ranking
        if len(ways) > 1:
            if prompt_key == "q3_dspy":
                pair_ids_comb = ways[0][1] + ways[1][1]
                orders_comb = orders_one + orders_two
                logprobs_comb = logprobs_one + logprobs_two
                
            elif "q3_mean" in prompt_key:
                
                orders_comb, logprobs_comb = [], []
                for logprobs1, logprobs2 in zip(all_info_logprobs_one, all_info_logprobs_two):
                    
                    if keep_only_most_top:
                        # Add the same pair ID twice: once for each direction's judgment, since we already flipped the labels in extract_info_mean_q3
                        pair_ids_comb = ways[0][1] + ways[0][1]
                        first_way, second_way = extract_info_mean_q3(logprobs1, logprobs2, keep_only_most_top=keep_only_most_top)
                        
                        orders_comb.append(first_way[0])
                        orders_comb.append(second_way[0])
                        
                        logprobs_comb.append(first_way[1])
                        logprobs_comb.append(second_way[1])
                        
                    else:        
                        pair_ids_comb = ways[0][1]            
                        order, logprob = extract_info_mean_q3(logprobs1, logprobs2)
                        orders_comb.append(order)
                        logprobs_comb.append(logprob)
            else:
                raise ValueError("Invalid prompt key for Q3.")
        else:
            pair_ids_comb = ways[0][1]
            orders_comb = orders_one
            logprobs_comb = logprobs_one

        # Rank computation
        ranked_documents = bradley_terry_model(
            pair_ids_comb, orders_comb, logprobs_comb)
        true_order = [el["doc_id"] for el in cluster_data["eval_docs"]]
        ranking_indices = {doc_id: idx for idx, doc_id in enumerate(ranked_documents['doc_id'])}
        rank = [ranking_indices[doc_id] + 1 for doc_id in true_order]
        rank = [len(rank) - r + 1 for r in rank]  # Invert rank

        log_or_print(f"\033[95mLLM Rank:\n {rank}\033[0m", logger)
        if isinstance(users_rank, np.ndarray) and users_rank.size > 0:
            log_or_print(f"\033[95mUsers rank: {users_rank}\033[0m", logger)
        rank_data.append(rank)
        info_to_bradley_terry["pair_ids_comb"] = pair_ids_comb
        info_to_bradley_terry["orders_comb"] = orders_comb
        
        return
    
    # this will be executed for each model the user wants to evaluate
    def run_metric(
        self,
        tm_model_data_path,
        llm_models,
        q1_temp=0,
        q2_temp=0,
        q3_temp=0,
        custom_seed=1234,
        do_both_ways=False,
        q1_q3_prompt_mode="q1_then_q3_dspy",
        q1_q2_prompt_mode="q1_then_q2_dspy",
        openai_key=None,
    ):  
        """
        Run "Proxann" metrics for a given topic model (full or set of topics) based on one or more LLMs.
        
        Parameters
        ----------
        tm_model_data_path : str
            Path to the topic modeling data.
        llm_models : list
            List of LLM models to use.
        temperatures : float, optional
            Temperatures value for the LLM generation in Q1/Q2/Q3, separated by commas.
        custom_seed : int, optional
            Custom seed for the LLM, by default None.
        do_both_ways : bool, optional
            Whether to run Q3 twice: once with A as the first document, then reversed, by default False.
        q1_q3_prompt_mode : str, optional
            Prompting mode for Q3, by default "q1_then_q3_dspy".
        q1_q2_prompt_mode : str, optional
            Prompting mode for Q2, by default "q1_then_q2_dspy".
        openai_key : str, optional
            OpenAI API key, by default None.
            
        Returns
        -------
        pd.DataFrame
            Dataframe with correlation results.
        """
        
        # Load topic modeling data with information for each topic being evaluated and normalize the keys
        tm_model_data = load_config_pilot(tm_model_data_path)
        
        # tm_model_data is a dictionary with keys representing the model path
        # we retain the inner dictionary, where each key represents a topic
        tm_model_data = tm_model_data[list(tm_model_data.keys())[0]]
        
        llm_results_q1, llm_results_q2, llm_results_q3, all_info_bradley_terry = [], [], [], []
        # ---------------------------------------------------------
        # For each topic ...
        # ---------------------------------------------------------

        # each cluster_data is the information for a topic
        for cluster_id, cluster_data in tm_model_data.items():
            self._logger.info(f"Cluster: {cluster_id}")
            
            rank_data = []
            info_to_bradley_terry = defaultdict(list)
            fit_data = [] 
            categories = []
            
            for llm_model in llm_models:
                # Create prompter for the LLM
                prompter = Prompter(model_type=llm_model, seed=custom_seed, openai_key=openai_key)
                # ----------------------------------------------
                # Q1_THEN_Q3
                # ----------------------------------------------
                self._logger.info("-- Executing Q1 / Q3...")
                # ==============================================
                # Q1
                # ==============================================                
                self.do_q1(
                    prompter=prompter, 
                    cluster_data=cluster_data, 
                    users_cats=[], 
                    categories=categories, 
                    temperature=q1_temp
                )
                
                # ==============================================
                # Q3
                # ==============================================
                # TODO: Add logic for when category is not category[-1] but each of the users' categories
                category = categories[-1]
                self.do_q3(
                    prompter=prompter,
                    prompt_mode=q1_q3_prompt_mode,
                    cluster_data=cluster_data,
                    rank_data=rank_data,
                    info_to_bradley_terry=info_to_bradley_terry,
                    users_rank=[],
                    category=category,
                    temperature=q3_temp,
                    doing_both_ways=do_both_ways
                )

                # ----------------------------------------------
                # Q1_THEN_Q2
                # ----------------------------------------------
                self._logger.info("-- Executing Q1 / Q2...")
                labels = self.do_q2(
                    prompter=prompter, 
                    prompt_mode=q1_q2_prompt_mode, 
                    cluster_data=cluster_data, 
                    fit_data=fit_data, 
                    category=category, 
                    temperature=q2_temp
                )
            
            llm_results_q1.append({
                "id": cluster_id,
                "n_annotators": len(llm_models),
                "annotators": llm_models,
                "categories": categories,
            })

            if fit_data != []:
                llm_results_q2.append({
                    "id": cluster_id,
                    "n_annotators": len(llm_models),
                    "annotators": llm_models,
                    "labels": labels,
                    "fit_data": [fit_data],
                })

            if rank_data != []:
                llm_results_q3.append({
                    "id": cluster_id,
                    "n_annotators": len(llm_models),
                    "annotators": llm_models,
                    "rank_data": rank_data
                })
                
            if info_to_bradley_terry:
                all_info_bradley_terry.append({
                    "id": cluster_id,
                    "n_annotators": len(llm_models),
                    "annotators": llm_models,
                    "info": info_to_bradley_terry,
                })
                
        llm_results_q2 = sorted(llm_results_q2, key=lambda x: x["id"])
        llm_results_q3 = sorted(llm_results_q3, key=lambda x: x["id"])
        
        corr_data = self.compute_llm_tm_corrs(tm_model_data, llm_results_q3, llm_results_q2)
                
        return corr_data, info_to_bradley_terry
    
    def compute_llm_tm_corrs(
        self,
        tm_model_data: dict,
        rank_llm_data: list,
        fit_llm_data: list,
    )-> pd.DataFrame:
        """
        Compute correlation between LLM and TM data.
        
        Parameters
        ----------
        tm_model_data : dict
            Topic modeling data.
        rank_llm_data : list
            List of rank data for the LLM.
        fit_llm_data : list
            List of fit data for the LLM.
            
        Returns
        -------
        pd.DataFrame
            Dataframe with correlation results.
        """
        
        model_data = []
        
        for topic_id, topic_data in tm_model_data.items():
            
            prob_data = np.array([doc["prob"] for doc in topic_data["eval_docs"]])

            assign_data = np.array([doc["assigned_to_k"] for doc in topic_data["eval_docs"]])
            
            model_data.append({
                "id": topic_id,
                "prob_data": prob_data,
                "assign_data": assign_data
            })   
            
        model_data = sorted(model_data, key=lambda x: x["id"])
        
        assert [k["id"] for k in model_data] == [k["id"] for k in rank_llm_data] == [k["id"] for k in fit_llm_data], "IDs do not match"
    
        corr_results = []
        for d_model, d_r_llm, d_f_llm in zip(model_data, rank_llm_data, fit_llm_data):
            # from the tm model
            prob_data = d_model["prob_data"]
            assign_data = d_model["assign_data"]
            
            # from the LLMs        
            annotator_results = {}
            r_llm = d_r_llm["rank_data"]
            r_annotators = d_r_llm["annotators"]
            f_llm = d_f_llm["fit_data"]
            f_annotators = d_f_llm["annotators"]
            
            # f_annotators should be the same as r_annotators
            assert r_annotators == f_annotators, "Annotators do not match"
            
            for a, r_llm_a, f_llm_a in zip(r_annotators, r_llm, f_llm):
                
                rank_rho_gt, _ = spearmanr(prob_data, r_llm_a)
                rank_tau_gt, _ = kendalltau(prob_data, r_llm_a)
                rank_ndcg_gt = ndcg_score([r_llm_a], [prob_data])
                fit_tau_tm, _ = kendalltau(prob_data, f_llm_a)
                fit_agree_tm = np.mean(assign_data == f_llm_a)
                
                annotator_results[f"rank_rho_tm_{a}"] = rank_rho_gt
                annotator_results[f"rank_tau_tm_{a}"] = rank_tau_gt
                annotator_results[f"rank_ndcg_tm_{a}"] = rank_ndcg_gt
            
                annotator_results[f"fit_tau_tm_{a}"] = fit_tau_tm
                annotator_results[f"fit_agree_tm_{a}"] = fit_agree_tm
                
            corr_results.append({
                "id": d_model["id"],
                **annotator_results
            })
            
        return pd.DataFrame(corr_results)