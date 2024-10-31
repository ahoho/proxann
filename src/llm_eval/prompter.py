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
            'chatgpt-4o-latest', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo',
            'gpt-4-32k', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview',
            'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106',
            'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301'
        ]
        
        self.OLLAMA_MODELS = [
            'llama3.2'
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
        category: str = None
    ) -> Union[str, List[str]]:
        
        def handle_q1(text: dict, topk: int = 10, num_words: int = 100) -> str:
            docs = "\n".join(f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
            keys = " ".join(text["topic_words"][:topk])
            return self._load_template("src/llm_eval/prompts/q1/question_prompt.txt").format(keys, docs)

        def handle_q2(text: dict, topk: int = 10) -> List[str]:
            keys = " ".join(text["topic_words"][:topk])
            template = self._load_template("src/llm_eval/prompts/q2/question_prompt.txt")
            return [template.format(keys, doc["text"]) for doc in text["eval_docs"]]

        def handle_q3(text: dict, cat: str, num_words: int = 100) -> List[str]:
            template = self._load_template("src/llm_eval/prompts/q3/question_prompt.txt")

            doc_pairs = []
            pair_ids = []

            # We create pairs of documents. In each pair, there is a 'DOCUMENT A' and a 'DOCUMENT B'. We keep track of the original IDs of the documents in pair_ids, which is a list of dictionaries with keys 'A' and 'B', each with the original ID of the document.
            for d1, d2 in itertools.combinations(text["eval_docs"], 2):
                # Shuffle the documents in the pair
                docs = [d1, d2]
                random.shuffle(docs)
                
                pair_ids.append({"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                
                doc_a = re.sub(r'^ID\d+\.', 'DOCUMENT A.', f"ID{docs[0]['doc_id']}. {extend_to_full_sentence(docs[0]['text'], num_words)}")
                doc_b = re.sub(r'^ID\d+\.', 'DOCUMENT B.', f"ID{docs[1]['doc_id']}. {extend_to_full_sentence(docs[1]['text'], num_words)}")
                
                doc_pairs.append(f"\n{doc_a}\n{doc_b}")

            return [template.format(cat, pair) for pair in doc_pairs], pair_ids
        
        def handle_q1_q3(text: dict, topk: int = 10, num_words: int = 100):
            docs = "\n".join(f"- {extend_to_full_sentence(doc['text'], num_words)}" for doc in text["exemplar_docs"])
            keys = " ".join(text["topic_words"][:topk])
            
            doc_pairs = []
            pair_ids = []

            # We create pairs of documents. In each pair, there is a 'DOCUMENT A' and a 'DOCUMENT B'. We keep track of the original IDs of the documents in pair_ids, which is a list of dictionaries with keys 'A' and 'B', each with the original ID of the document.
            for d1, d2 in itertools.combinations(text["eval_docs"], 2):
                # Shuffle the documents in the pair
                docs = [d1, d2]
                random.shuffle(docs)
                
                pair_ids.append({"A": docs[0]["doc_id"], "B": docs[1]["doc_id"]})
                
                doc_a = re.sub(r'^ID\d+\.', 'DOCUMENT A.', f"ID{docs[0]['doc_id']}. {extend_to_full_sentence(docs[0]['text'], num_words)}")
                doc_b = re.sub(r'^ID\d+\.', 'DOCUMENT B.', f"ID{docs[1]['doc_id']}. {extend_to_full_sentence(docs[1]['text'], num_words)}")
                
                doc_pairs.append(f"\n{doc_a}\n{doc_b}")
            
            template = self._load_template("src/llm_eval/prompts/q1_q3/question_prompt.txt")
            
            return [template.format(keys,docs,pair) for pair in doc_pairs], pair_ids

        question_handlers = {
            "q1": handle_q1,
            "q2": handle_q2,
            "q3": lambda text: handle_q3(text, category),
            "q1_q3": handle_q1_q3
        }

        if question_type not in question_handlers:
            raise ValueError("Invalid question type")
        
        return question_handlers[question_type](text_for_prompt)