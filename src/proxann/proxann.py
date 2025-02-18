import configparser
import json
import logging
import sys
import numpy as np
import pandas as pd # type: ignore
import pathlib
from typing import Optional

from scipy import sparse
from src.proxann.prompter import Prompter
from src.proxann.utils import bradley_terry_model, extract_info_binary_q2, extract_info_q1_q3, extract_logprobs
from src.user_study_data_collector.jsonify.topic_json_formatter import TopicJsonFormatter
from src.user_study_data_collector.topics_docs_selection.topic_selector import TopicSelector
from src.utils.utils import init_logger, load_vocab_from_txt, log_or_print

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
        
        self._logger.info("ProxAnn object initialized.")
        
        return
    
    def generate_user_provided_json(
        self,
        path_user_study_config_file: str,
        output_path: str = None,
    ) -> int:
        """Generate a JSON file with user-provided model data.
        
        We assume the user gives the <path_user_study_config_file> already formatted or that the backend generates it accordingly based on the files uploaded by the user.
        
        Parameters
        ----------
        path_user_study_config_file : str
            Path to the user study configuration file.
        output_path : str
            Path to save the JSON file.
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

        log_or_print(f"Selected topics: {selected_topics}", self._logger)

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
        with open(output_path, 'w') as file:
            json.dump(combined_out, file, indent=4)    
        
        return ""
    
    # =========================================================================
    # Logic for Q1 / Q2 / Q3
    # =========================================================================
    def do_q1(
        prompter: Prompter,
        cluster_data: dict,
        users_cats: list,
        categories: list,
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
        dft_system_prompt : str, optional
            Default system prompt for Q1, by default "src/proxann/prompts/q1/simplified_system_prompt.txt".
        logging : logging.Logger, optional
            Logger object, by default None
        """
        log_or_print("Executing Q1...", logger)

        question = prompter.get_prompt(cluster_data, "q1")
        category, _ = prompter.prompt(
            dft_system_prompt, question, use_context=False)  # max_tokens=10

        categories.append(category)
        log_or_print(f"\033[92mUser categories: {users_cats}\033[0m", logger)
        log_or_print(f"\033[94mModel category: {category}\033[0m", logger)

        return


    def do_q2(
        prompter: Prompter,
        prompt_mode: str,
        llm_model: str,
        cluster_data: dict,
        fit_data: list,
        category: str,
        user_cats: list = None,
        dft_system_prompt: str = "src/proxann/prompts/XXX/simplified_system_prompt.txt",
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
        llm_model : str
            LLM model to use.
        cluster_data : dict
            Information for a topic as given by the data loaded from args.tm_model_data_path.
        fit_data : list
            List to store the fit scores.
        category : str
            LLM-generated category.
        user_cats : list, optional
            List of user-generated categories during the user study, by default None.
        dft_system_prompt : str, optional
            Default system prompt for Q2, by default "src/proxann/prompts/XXX/simplified_system_prompt.txt".
        use_context : bool, optional
            Whether to use the context for the prompt, by default False.

        Returns
        -------
        list
            List of user categories.
        """
        if prompt_mode == "q1_then_q2_dspy":
            if "llama" in llm_model:
                prompt_key = "q2_dspy_llama"
            elif "qwen" in llm_model:
                prompt_key = "q2_dspy_qwen"
            else:
                prompt_key = "q2_dspy"
            log_or_print(f"Using prompt key: {prompt_key}", logger)
            questions = prompter.get_prompt(cluster_data, prompt_key, category)
        else:
            do_q2_with_q1_fixed = prompt_mode == "q1_then_q2_fix_cat"
            questions = prompter.get_prompt(
                cluster_data, "binary_q2", category, do_q2_with_q1_fixed=do_q2_with_q1_fixed)

        if "dspy" in prompt_mode:
            dft_system_prompt = None

        if user_cats:
            labels = user_cats * len(questions)

            #  we do not to make one prompt per user category (each user has a different category), and we want to use each user's category to determine the fit score for each document in the evaluation set
            for cat in user_cats:
                if prompt_mode == "q1_then_q2_dspy":
                    if "llama" in llm_model:
                        prompt_key = "q2_dspy_llama"
                    elif "qwen" in llm_model:
                        prompt_key = "q2_dspy_qwen"
                    else:
                        prompt_key = "q2_dspy"
                    log_or_print(f"Using prompt key: {prompt_key}", logger)
                    questions = prompter.get_prompt(cluster_data, prompt_key, cat)
                else:
                    do_q2_with_q1_fixed = prompt_mode == "q1_then_q2_fix_cat"
                    questions = prompter.get_prompt(
                        cluster_data, "binary_q2", cat, do_q2_with_q1_fixed=do_q2_with_q1_fixed)

                for question in questions:
                    response_q2, _ = prompter.prompt(
                        dft_system_prompt, question, use_context=use_context)
                    score = extract_info_binary_q2(response_q2)
                    log_or_print(f"\033[92mFit: {score}\033[0m", logger)
                    fit_data.append(score)
        else:
            labels = [category] * len(questions)
        for question in questions:
            response_q2, _ = prompter.prompt(
                dft_system_prompt, question, use_context=use_context)
            log_or_print(f"\033[96mFit: {response_q2}\033[0m", logger)
            score = extract_info_binary_q2(response_q2)
            #if "marginally" in response_q2.lower() or "marginal" in response_q2.lower() or "maybe" in response_q2.lower():
            #if "no" in response_q2.lower():
            #    import pdb; pdb.set_trace()
            log_or_print(f"\033[92mFit: {score}\033[0m", logger)
            fit_data.append(score)

        return labels


    def do_q3(
        prompter: Prompter,
        prompt_mode: str,
        llm_model: str,
        cluster_data: dict,
        rank_data: list,
        users_rank: list,
        category: str,
        doing_both_ways: bool = False,
        dft_system_prompt: str = "src/proxann/prompts/q3/simplified_system_prompt.txt",
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
        llm_model : str
            LLM model to use.
        cluster_data : dict
            Information for a topic as given by the data loaded from args.tm_model_data_path.
        rank_data : list
            List to store the rank data.
        users_rank : list
            List of user ranks.
        category : str
            LLM-generated category.
        doing_both_ways : bool, optional
            Whether to run Q3 twice: once with A as the first document, then reversed, by default False.
        dft_system_prompt : str, optional
            Default system prompt for Q3, by default "src/proxann/prompts/q3/simplified_system_prompt.txt".
        use_context : bool, optional
            Whether to use the context for the prompt, by default False.
        logger : logging.Logger, optional
            Logger object, by default None.
        """
        # if "dspy" in prompt_mode:
        #   dft_system_prompt = None

        do_q3_with_q1_fixed = prompt_mode == "q1_then_q3_fix_cat"

        if prompt_mode == "q1_then_q3_dspy":
            prompt_key = "q3_dspy_llama" if "llama" in llm_model else "q3_dspy"
            log_or_print(f"-- Using prompt key: {prompt_key}", logger)
        else:
            prompt_key = "q3"
        q3_out = prompter.get_prompt(cluster_data, prompt_key, category=category, do_q3_with_q1_fixed=do_q3_with_q1_fixed, doing_both_ways=doing_both_ways)

        if isinstance(q3_out, tuple) and len(q3_out) > 2:  # Both ways
            questions_one, pair_ids_one, questions_two, pair_ids_two = q3_out
            ways = [[questions_one, pair_ids_one], [questions_two, pair_ids_two]]
        else:  # Single way
            questions, pair_ids = q3_out
            ways = [[questions, pair_ids]]

        labels_one, orders_one, rationales_one, logprobs_one = [], [], [], []
        labels_two, orders_two, rationales_two, logprobs_two = [], [], [], []

        for way_id, (questions, pair_ids) in enumerate(ways):
            log_or_print(
                f"-- Executing Q3 ({'both ways' if len(ways) > 1 else 'one way'})...", logger)
            for question in questions:
                pairwise, pairwise_logprobs = prompter.prompt(
                    dft_system_prompt, question, use_context=use_context
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
                        log_or_print(f"\033[92mOrder: {order}\033[0m", logger)
                    elif len(ways) > 1 and way_id == 1:
                        labels_two.append(label)
                        orders_two.append(order)
                        rationales_two.append(rationale)
                        logprobs_two.append(extract_logprobs(
                            pairwise_logprobs, prompter.backend, logger))
                        log_or_print(f"\033[94mOrder: {order}\033[0m", logger)
                    else:
                        labels_one.append(label)
                        orders_one.append(order)
                        rationales_one.append(rationale)
                        logprobs_one.append(extract_logprobs(
                            pairwise_logprobs, prompter.backend, logger))
                        log_or_print(f"\033[92mOrder: {order}\033[0m", logger)
                except Exception as e:
                    log_or_print(
                        f"-- Error extracting info from prompt: {e}", "error", logger)

        # Combine results for ranking
        if len(ways) > 1:
            pair_ids_comb = ways[0][1] + ways[1][1]
            orders_comb = orders_one + orders_two
            logprobs_comb = logprobs_one + logprobs_two
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
        log_or_print(f"\033[95mUsers rank: {users_rank}\033[0m", logger)
        rank_data.append(rank)

        return