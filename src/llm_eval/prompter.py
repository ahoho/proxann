import json
import logging
import os
import random
import re
import time
import pprint
import itertools
from typing import Union, List
from dotenv import load_dotenv
import requests
import ollama  # type: ignore
from openai import OpenAI

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
        
        # Determine backend based on model_type
        if model_type in self.GPT_MODELS:
            load_dotenv(path_open_ai_key)
            self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            self.llama_cpp_host = llama_cpp_host
            self.backend = "llama_cpp"
            self._logger.info(
                f"-- -- Using llama_cpp API with host: {llama_cpp_host}"
            )
        else:
            raise ValueError("Unsupported model_type specified.")

    def _load_template(self, template_path: str) -> str:
        with open(template_path, 'r') as file:
            return file.read()

    def prompt(
        self,
        system_prompt_template_path: str,
        question: str,
        use_context: bool = False
    ) -> Union[str, List[str]]:
        template = self._load_template(system_prompt_template_path)
        context_value = self.context if self.context and use_context else ""
        start_time = time.time()

        try:
            if self.model_type in self.GPT_MODELS:
                response = self.open_ai_client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": template},
                        {"role": "user", "content": question}
                    ],
                    stream=False,
                    temperature=self.params['temperature'],
                    max_tokens=self.params.get('max_tokens', 1000),
                    logprobs=True,
                    top_logprobs=10,
                )
                result = response.choices[0].message.content
                logprobs = response.choices[0].logprobs.content
                self.context = result if use_context else None
            else:
                
                if self.backend == "ollama":
                    response = ollama.generate(
                        system=template,
                        prompt=question,
                        model=self.model_type,
                        stream=False,
                        options=self.params,
                        context=context_value
                    )
                    result = response['response']
                    logprobs = None
                    self.context = response["context"]
                    
                elif self.backend == "llama_cpp":
                    payload = {
                        "messages": [
                            {"role": "system", "content": template},
                            {"role": "user", "content": question}
                        ],
                        "temperature": self.params.get("temperature", 0.7),
                        "max_tokens": self.params.get("max_tokens", 100),
                        "logprobs": 1,  
                        "n_probs": 1
                    }
                    response = requests.post(self.llama_cpp_host, json=payload)
                    response_data = response.json()

                    if response.status_code == 200:
                        result = response_data["choices"][0]["message"]["content"]
                        logprobs = response_data.get("completion_probabilities", [])

                        self.context = result if use_context else None
                    else:
                        raise RuntimeError(f"llama_cpp API error: {response_data.get('error', 'Unknown error')}")
                    
            pprint.pprint(result)
            print(f"Prompt time: {format(time.time() - start_time, '.2f')} seconds")
            return result, logprobs

        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return "Error while processing the request."

    def get_prompt(
        self,
        text_for_prompt: dict,
        question_type: str,
        category: str = None,
        nr_few_shot: int = 1,
        doing_both_ways: bool = True,
        do_q3_with_q1_fixed: bool = False,
        path_examples: str = "src/llm_eval/prompts/few_shot_examples.json",
        base_prompt_path: str = "src/llm_eval/prompts/"
    ) -> Union[str, List[str]]:

        # Read JSON with few-shot examples
        with open(path_examples, 'r') as file:
            few_shot_examples = json.load(file)
        
        def load_template_path(q_type: str) -> str:
            """Helper function to construct the template path based on question type."""
            return f"{base_prompt_path}{q_type}/instructions_question_prompt.txt"

        def handle_q1(text: dict, few_shot: dict, topk: int = 10, num_words: int = 100, nr_few_shot=1) -> str:
            
            # Few-shot examples
            examples_q1 = few_shot["q1"][:nr_few_shot]
            examples = [
                "KEYWORDS: {}\nDOCUMENTS:\n {}\nCATEGORY: {}".format(
                    ex['example']['keywords'],
                    ''.join(f"- {doc}\n" for doc in ex['example']['documents']),
                    ex['example']['response']['label']
                )
                for ex in examples_q1
            ]

            # Actual question to the LLM (keys and docs) 
            docs = "\n".join(f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
            keys = " ".join(text["topic_words"][:topk])
            template_path = load_template_path("q1")
            return self._load_template(template_path).format("\n".join(examples), keys, docs)

        def handle_q2(text: dict, few_shot: dict, cat: str, topk: int = 10, nr_few_shot=1) -> List[str]:
            
            # Few shot examples
            examples_q2 = few_shot["q2"][:nr_few_shot]
            examples = "\n".join([
                #"CATEGORY: {}\nDOCUMENT: {}\nRESPONSE:\nSCORE: {}\n RATIONALE: {}".format(
                "CATEGORY: {}\nDOCUMENT: {}\nSCORE: {}\n RATIONALE: {}".format(
                    cat,
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
                    "KEYWORDS: {}\nEXEMPLAR DOCUMENTS:\n {}\nCATEGORY: {}\nEVAL DOCUMENTS: \n- DOCUMENT A: {} \n- DOCUMENT B:{}\nCLOSEST: {} \nRATIONALE: {}".format(
                        ex1['example']['keywords'],
                        ''.join(f"- {doc}\n" for doc in ex1['example']['documents']),
                        ex1['example']['response']['label'],
                        ex3['example']['documents']["A"],
                        ex3['example']['documents']["B"],
                        ex3['example']['response']["order"],
                        ex3['example']['response']["rationale"] + "\n",
                    )
                    for ex1, ex3 in zip(examples_q1, examples_q3)
                ])
                exemplar_docs = "\n".join(f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
                keys = " ".join(text["topic_words"][:topk])
            else:
                examples = "\n".join([
                    "CATEGORY: {}\nDOCUMENTS: \n- DOCUMENT A: {} \n- DOCUMENT B:{}\nCLOSEST: {} \nRATIONALE: {}".format(
                        ex['example']['category'],
                        ex['example']['documents']["A"],
                        ex['example']['documents']["B"],
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
                    [template.format(examples, keys, exemplar_docs, cat, pair) for pair in doc_pairs_one],
                    pair_ids_one,
                    [template.format(examples, keys, exemplar_docs, cat, pair) for pair in doc_pairs_two],
                    pair_ids_two
                )
            else:
                return [template.format(examples, keys, exemplar_docs, cat, pair) for pair in doc_pairs_one], pair_ids_one

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
            "q1": lambda text: handle_q1(text, few_shot_examples, nr_few_shot=nr_few_shot),
            "q2": lambda text: handle_q2(text, few_shot_examples, category,nr_few_shot=nr_few_shot),
            "q3": lambda text: handle_q3(text, few_shot_examples, category, nr_few_shot=nr_few_shot, doing_both_ways=doing_both_ways, do_q3_with_q1_fixed=do_q3_with_q1_fixed),
            "q1_q2": lambda text: handle_q1_q2(text, few_shot_examples, nr_few_shot=nr_few_shot),
            "q1_q3": lambda text: handle_q1_q3(text, few_shot_examples, nr_few_shot=nr_few_shot)
        }

        if question_type not in question_handlers:
            raise ValueError("Invalid question type")
        
        return question_handlers[question_type](text_for_prompt)
