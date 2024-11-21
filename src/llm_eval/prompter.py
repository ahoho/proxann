import json
import logging
import os
import random
import re
import itertools
from typing import Union, List
from dotenv import load_dotenv
from joblib import Memory
import requests
import ollama  # type: ignore
from openai import OpenAI
memory = Memory(location='cache', verbose=0)

from src.llm_eval.utils import extend_to_full_sentence

class Prompter:
    def __init__(
        self,
        model_type: str = "ollama",
        temperature: float = 0.2,
        top_p: float = 0.1,
        random_seed: int = 1234,
        frequency_penalty: float = 0.0,
        path_open_ai_key: str = ".env",
        ollama_host: str = "http://kumo01.tsc.uc3m.es:11434",
        llama_cpp_host: str = "http://kumo01:11435/v1/chat/completions",
        logger=None
    ):  
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
        
        self.GPT_MODELS = [
            'gpt-4o-2024-08-06',
            'gpt-4o-mini-2024-07-18',
            'chatgpt-4o-latest', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o',
            'gpt-4-32k', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview',
            'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106',
            'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301'
        ]
        
        self.OLLAMA_MODELS = [
            'llama3.2',
            'llama3.1:8b-instruct-q8_0'
        ]
        
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "frequency_penalty": frequency_penalty
        }    
        
        self.model_type = model_type
        self.context = None
        self.llama_cpp_host = llama_cpp_host
        
        # Determine backend based on model_type
        if model_type in self.GPT_MODELS:
            load_dotenv(path_open_ai_key)
            # self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.backend = "openai"
            self._logger.info(
                f"-- -- Using OpenAI API with model: {model_type}")
        elif model_type in self.OLLAMA_MODELS:
            os.environ['OLLAMA_HOST'] = ollama_host
            self.backend = "ollama"
            self._logger.info(
                f"-- -- Using OLLAMA API with host: {ollama_host}"
            )
        elif model_type == "llama_cpp":
            self.backend = "llama_cpp"
            self._logger.info(
                f"-- -- Using llama_cpp API with host: {llama_cpp_host}"
            )
        else:
            raise ValueError("Unsupported model_type specified.")

    def _load_template(self, template_path: str) -> str:
        with open(template_path, 'r') as file:
            return file.read()

    @staticmethod
    @memory.cache
    def _cached_prompt_impl(
        template: str,
        question: str,
        model_type: str,
        backend: str,
        params: tuple,
        context=None,
        use_context: bool = False,
    ) -> dict:
        """Caching setup."""
        
        print("Cache miss: computing results...")
        if backend == "openai":
            result, logprobs = Prompter._call_openai_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),  
            )
        elif backend == "ollama":
            result, logprobs, context = Prompter._call_ollama_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                context=context,
            )
        elif backend == "llama_cpp":
            result, logprobs = Prompter._call_llama_cpp_api(
                template=template,
                question=question,
                params=dict(params), 
            )
        else:
            raise ValueError(f"-- -- Unsupported backend: {backend}")

        return {
            "inputs": {
                "template": template,
                "question": question,
                "model_type": model_type,
                "backend": backend,
                "params": dict(params),
                "context": context if use_context else None,
                "use_context": use_context,
            },
            "outputs": {
                "result": result,
                "logprobs": logprobs,
            },
        }


    @staticmethod
    def _call_openai_api(template, question, model_type, params):
        """Handles the OpenAI API call."""
        
        open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = open_ai_client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ],
            stream=False,
            temperature=params["temperature"],
            max_tokens=params.get("max_tokens", 1000),
            logprobs=True,
            top_logprobs=10,
        )
        result = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return result, logprobs


    @staticmethod
    def _call_ollama_api(template, question, model_type, params, context):
        """Handles the OLLAMA API call."""
        response = ollama.generate(
            system=template,
            prompt=question,
            model=model_type,
            stream=False,
            options=params,
            context=context,
        )
        result = response["response"]
        logprobs = None
        context = response.get("context", None)
        return result, logprobs, context

    @staticmethod
    def _call_llama_cpp_api(template, question, params, llama_cpp_host="http://kumo01:11435/v1/chat/completions"):
        """Handles the llama_cpp API call."""
        payload = {
            "messages": [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 100),
            "logprobs": 1,
            "n_probs": 1,
        }
        response = requests.post(llama_cpp_host, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            result = response_data["choices"][0]["message"]["content"]
            logprobs = response_data.get("completion_probabilities", [])
        else:
            raise RuntimeError(f"llama_cpp API error: {response_data.get('error', 'Unknown error')}")

        return result, logprobs

    def prompt(
        self,
        system_prompt_template_path: str,
        question: str,
        use_context: bool = False,
    ) -> Union[str, List[str]]:
        """Public method to execute a prompt given a system prompt template and a question."""

        # Load the system prompt template
        with open(system_prompt_template_path, "r") as file:
            system_prompt_template = file.read()

        # Ensure hashable params for caching and get cached data / execute prompt
        params_tuple = tuple(sorted(self.params.items()))
        cached_data = self._cached_prompt_impl(
            template=system_prompt_template,
            question=question,
            model_type=self.model_type,
            backend=self.backend,
            params=params_tuple,
            context=self.context if use_context else None,
            use_context=use_context,
        )

        result = cached_data["outputs"]["result"]
        logprobs = cached_data["outputs"]["logprobs"]

        # Update context if necessary
        if use_context:
            self.context = cached_data["inputs"]["context"]
            
        return result, logprobs

    def get_prompt(
        self,
        text_for_prompt: dict,
        question_type: str,
        category: str = None,
        nr_few_shot: int = 2,
        doing_both_ways: bool = True,
        do_q3_with_q1_fixed: bool = False,
        do_q2_with_q1_fixed: bool = False,
        generate_description: bool = False,
        path_examples: str = "src/llm_eval/prompts/few_shot_examples.json",
        base_prompt_path: str = "src/llm_eval/prompts/"
    ) -> Union[str, List[str]]:

        # Read JSON with few-shot examples
        with open(path_examples, 'r') as file:
            few_shot_examples = json.load(file)
        
        def load_template_path(q_type: str) -> str:
            """Helper function to construct the template path based on question type."""
            return f"{base_prompt_path}{q_type}/instructions_question_prompt.txt"

        def handle_q1(text: dict, few_shot: dict, topk: int = 10, num_words: int = 100, nr_few_shot=1, generate_description:bool=False) -> str:
            
            # Few-shot examples
            examples_q1 = few_shot["q1"][:nr_few_shot]
            
            if generate_description:
                examples = [
                    "KEYWORDS: {}\nDOCUMENTS: {}\nCATEGORY: {}\nDESCRIPTION:{}".format(
                        ex['example']['keywords'],
                        ''.join(f"\n- {doc}" for doc in ex['example']['documents']),
                        ex['example']['response']['label'],
                        ex['example']['response']['description']
                    )
                    for ex in examples_q1
                ]
            else:            
                examples = [
                    "KEYWORDS: {}\nDOCUMENTS: {}\nCATEGORY: {}".format(
                        ex['example']['keywords'],
                        ''.join(f"\n- {doc}" for doc in ex['example']['documents']),
                        ex['example']['response']['label']
                    )
                    for ex in examples_q1
                ]

            # Actual question to the LLM (keys and docs) 
            docs = "".join(f"\n- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
            keys = " ".join(text["topic_words"][:topk])
            template_path = load_template_path("q1_with_desc") if generate_description else load_template_path("q1")
            
            return self._load_template(template_path).format("\n".join(examples), keys, docs)

        def handle_q2(text: dict, few_shot: dict, cat: str, topk: int = 10, nr_few_shot=1) -> List[str]:
            
            # Few shot examples
            examples_q2 = few_shot["q2"][:nr_few_shot]
            examples = "\n".join([
                #"CATEGORY: {}\nDOCUMENT: {}\nRESPONSE:\nSCORE: {}\n RATIONALE: {}".format(
                "CATEGORY: {}\nDOCUMENT: {}\nSCORE: {}\n RATIONALE: {}".format(
                    ex['example']['category'],
                    ex['example']['document'],
                    ex['example']['response']['score'],
                    ex['example']['response']['rationale']
                )
                for ex in examples_q2
            ])
            
            # Actual question to the LLM (category from Q1 and doc)
            keys = " ".join(text["topic_words"][:topk])
            template_path = load_template_path("q2")
            return [self._load_template(template_path).format(examples, keys, doc["text"]) for doc in text["eval_docs"]]

        def handle_binary_q2(text: dict, few_shot: dict, cat: str, topk: int = 10, nr_few_shot=2, fit_threshold=4, num_words: int = 100, do_q2_with_q1_fixed: bool=False) -> List[str]:
                        
            # Few shot examples
            nr_few_shot = 1 if do_q2_with_q1_fixed else nr_few_shot
            examples_q1 = few_shot.get("q1", [])[:nr_few_shot] if do_q2_with_q1_fixed else []
            examples_q2 = few_shot["q2"][:nr_few_shot]
            
            if do_q2_with_q1_fixed and examples_q1:
                # here we will always have 1 example (otherwise it would get too long), so we do not add delimiters for the examples
                examples = "\n".join([
                    "KEYWORDS: {}\nEXEMPLAR DOCUMENTS: {}\n\nEVALUATION DOCUMENT: {} \nCATEGORY: {}\nFIT: {} \n".format(
                        ex1['example']['keywords'],
                        ''.join(f"\n- {doc}" for doc in ex1['example']['documents']),
                        ex2['example']['document'],
                        ex1['example']['response']['label'],
                        "YES" if ex2['example']['response']['score'] > fit_threshold else "NO" + "\n"
                    )
                    for ex1, ex2 in zip(examples_q1, examples_q2)
                ])
                exemplar_docs = "".join(f"\n- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
                keys = " ".join(text["topic_words"][:topk])
            else:                       
                examples = "\n===\n".join([
                    "DOCUMENT: {}\nCATEGORY: {}\nFIT: {}".format(
                        ex['example']['document'],
                        ex['example']['category'],
                        "YES" if ex['example']['response']['score'] > fit_threshold else "NO",
                    )
                    for ex in examples_q2
                ])
                exemplar_docs, keys = "", ""
            
            # Actual question to the LLM (category from Q1 and doc)
            template_path = load_template_path("q2_fix_cat" if do_q2_with_q1_fixed else "binary_q2")
            if do_q2_with_q1_fixed:
                return [self._load_template(template_path).format(examples, keys, exemplar_docs, extend_to_full_sentence(doc['text'], num_words), cat) for doc in text["eval_docs"]]
            else:
                return [self._load_template(template_path).format(examples, extend_to_full_sentence(doc['text'], num_words),cat) for doc in text["eval_docs"]]
        
        def handle_q3(
            text: dict,
            few_shot: dict,
            cat: str,
            num_words: int = 100,
            topk: int = 10,
            nr_few_shot=1,
            doing_both_ways=True,
            do_q3_with_q1_fixed: bool = False
        ) -> List[str]:
            
            # Few-shot examples
            examples_q3 = few_shot["q3"][:nr_few_shot]
            examples_q1 = few_shot.get("q1", [])[:nr_few_shot] if do_q3_with_q1_fixed else []
            
            if do_q3_with_q1_fixed and examples_q1:
                examples = "\n".join([
                    "KEYWORDS: {}\nEXEMPLAR DOCUMENTS: {}\n\nEVALUATION DOCUMENTS: \n- DOCUMENT A: {} \n- DOCUMENT B:{} \nCATEGORY: {}\nCLOSEST: {} \n".format(
                        ex1['example']['keywords'],
                        ''.join(f"\n- {doc}" for doc in ex1['example']['documents']),
                        ex3['example']['documents']["A"],
                        ex3['example']['documents']["B"],
                        ex1['example']['response']['label'],
                        ex3['example']['response']["order"] + "\n"
                    )
                    for ex1, ex3 in zip(examples_q1, examples_q3)
                ])
                exemplar_docs = "".join(f"\n- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
                keys = " ".join(text["topic_words"][:topk])
            else:
                examples = "\n".join([
                    "DOCUMENTS: \n- DOCUMENT A: {} \n- DOCUMENT B:{}\nCATEGORY: {}\nRATIONALE: {} \nORDER: {}".format(
                        
                        ex['example']['documents']["A"],
                        ex['example']['documents']["B"],
                        ex['example']['category'],
                        ex['example']['response']["order"],
                        ex['example']['response']["rationale"] + "\n",
                    )
                    for ex in examples_q3
                ])
                exemplar_docs, keys = "", ""

            # Document pairs generation
            doc_pairs_one, doc_pairs_two = [], []
            pair_ids_one, pair_ids_two = [], []
            
            eval_docs = random.sample(text["eval_docs"], len(text["eval_docs"]))
            for d1, d2 in itertools.combinations(eval_docs, 2):
                docs_list = [[d1, d2], [d2, d1]] if doing_both_ways else [random.sample([d1, d2], 2)]
                
                for i, docs in enumerate(docs_list):
                    doc_a = re.sub(r'^ID\d+\.', '- DOCUMENT A.', f"ID{docs[0]['doc_id']}. {extend_to_full_sentence(docs[0]['text'], num_words)}")
                    doc_b = re.sub(r'^ID\d+\.', '- DOCUMENT B.', f"ID{docs[1]['doc_id']}. {extend_to_full_sentence(docs[1]['text'], num_words)}")
                    
                    if doing_both_ways:
                        if i % 2 == 0:
                            doc_pairs_one.append(f"\n{doc_a}\n{doc_b}")
                            pair_ids_one.append({"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                        else:
                            doc_pairs_two.append(f"\n{doc_a}\n{doc_b}")
                            pair_ids_two.append({"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                    else:
                        doc_pairs_one.append(f"\n{doc_a}\n{doc_b}")
                        pair_ids_one.append({"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})

            # Template loading and formatting
            template_path = load_template_path("q1_then_q3_fix_cat" if do_q3_with_q1_fixed else "q3")
            template = self._load_template(template_path)
            
            if doing_both_ways:
                return (
                    [template.format(examples, keys, exemplar_docs, pair, cat) for pair in doc_pairs_one],
                    pair_ids_one,
                    [template.format(examples, keys, exemplar_docs, pair, cat) for pair in doc_pairs_two],
                    pair_ids_two
                )
            else:
                return [template.format(examples, keys, exemplar_docs, pair, cat) for pair in doc_pairs_one], pair_ids_one

        def handle_q1_q2(text: dict, few_shot: dict, topk: int = 10, num_words: int = 100, nr_few_shot=1):
            docs = "\n".join(f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
            keys = " ".join(text["topic_words"][:topk])
            template_path = load_template_path("q1_q2")
            return [self._load_template(template_path).format(keys, docs, doc["text"]) for doc in text["eval_docs"]]
            
        def handle_q1_q3(text: dict, few_shot: dict, topk: int = 10, num_words: int = 100, nr_few_shot=1):
            docs = "\n".join(f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
            keys = " ".join(text["topic_words"][:topk])
            doc_pairs = []
            pair_ids = []
            for d1, d2 in itertools.combinations(text["eval_docs"], 2):
                docs = [d1, d2]
                random.shuffle(docs)
                pair_ids.append({"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                doc_a = re.sub(r'^ID\d+\.', 'DOCUMENT A.', f"ID{docs[0]['doc_id']}. {extend_to_full_sentence(docs[0]['text'], num_words)}")
                doc_b = re.sub(r'^ID\d+\.', 'DOCUMENT B.', f"ID{docs[1]['doc_id']}. {extend_to_full_sentence(docs[1]['text'], num_words)}")
                doc_pairs.append(f"\n{doc_a}\n{doc_b}")
            template_path = load_template_path("q1_q3")
            return [self._load_template(template_path).format(keys, docs, pair) for pair in doc_pairs], pair_ids

        question_handlers = {
            "q1": lambda text: handle_q1(text, few_shot_examples, nr_few_shot=nr_few_shot, generate_description=generate_description),
            "q2": lambda text: handle_q2(text, few_shot_examples, category,nr_few_shot=nr_few_shot),
            "binary_q2": lambda text: handle_binary_q2(text, few_shot_examples, category,nr_few_shot=nr_few_shot, do_q2_with_q1_fixed=do_q2_with_q1_fixed),
            "q3": lambda text: handle_q3(text, few_shot_examples, category, nr_few_shot=nr_few_shot, doing_both_ways=doing_both_ways, do_q3_with_q1_fixed=do_q3_with_q1_fixed),
            "q1_q2": lambda text: handle_q1_q2(text, few_shot_examples, nr_few_shot=nr_few_shot),
            "q1_q3": lambda text: handle_q1_q3(text, few_shot_examples, nr_few_shot=nr_few_shot)
        }

        if question_type not in question_handlers:
            raise ValueError("Invalid question type")
        
        return question_handlers[question_type](text_for_prompt)
