import logging
import os
import pathlib
from typing import Optional

import dspy
import pandas as pd
from dotenv import load_dotenv
from dspy.datasets import Dataset
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import train_test_split

from src.proxann.utils import extend_to_full_sentence

class ThetasDataset(Dataset):

    def __init__(
        self,
        data_fpath: str,
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        seed: Optional[int] = 11235,
        inputs = ["category", "document_a", "document_b"],
        outputs = ["closest"],
        mode ="q3",
        num_words_truncate: int = 100,
        *args,
        **kwargs
    ) -> None:
        """
        fact -> question
        """

        super().__init__(*args, **kwargs)

        self._train = []
        self._dev = []
        self._test = []

        # Read the training data (save as json)
        train_data = pd.read_json(data_fpath)
        
        if mode == "q3":
        
            # keep first category from each user
            train_data["closest"] = train_data.apply(lambda x: "A" if x["users_rank_winner"] == x["doc_id1"] else "B", axis=1)
            
            # truncate the documents to num_words_truncate
            train_data["document_a"] = train_data["doc1"].apply(lambda x: extend_to_full_sentence(x,num_words_truncate))
            train_data["document_b"] =  train_data["doc2"].apply(lambda x: extend_to_full_sentence(x,num_words_truncate))
            
            train_data = train_data[(inputs + outputs)]
        
        elif mode == "q2":
            train_data["category"] = train_data["user_label"]
            train_data["fit"] = train_data.apply(lambda x: "YES" if x["bin_fit"] == 1 else "NO", axis=1)
            train_data["document"] = train_data["doc"].apply(lambda x: extend_to_full_sentence(x,num_words_truncate))
            inputs = ["category", "document"]
            train_data = train_data[(inputs + ["fit"])]            
        else:
            raise ValueError("mode must be either 'q2' or 'q3'")
        
        train_data, temp_data = train_test_split(train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(*inputs) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(*inputs) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(*inputs) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')
        
################################################################################
# SIGNATURE & MODULE
################################################################################
class Q1Signature(dspy.Signature):
    """Analyze both the KEYWORDS and the content of the documents to create a clear, concise label that accurately reflects the overall theme they share."""
    KEYWORDS = dspy.InputField()
    DOCUMENTS = dspy.InputField()
    CATEGORY = dspy.OutputField(desc="Category that best describes the topic connecting all the documents")
    
class Q1Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.q1signature = dspy.Predict(Q1Signature)

    def forward(self, keywords: str, documents: str):
        category = self.q1signature(KEYWORDS=keywords, DOCUMENTS=documents).CATEGORY
        
        return dspy.Prediction(category = category)


class Q3Signature(dspy.Signature):
    """Determine which document is more closely related to the given category"""
    CATEGORY = dspy.InputField()
    DOCUMENT_A = dspy.InputField()
    DOCUMENT_B = dspy.InputField()
    CLOSEST = dspy.OutputField(desc="Document that is more closely related to the category (A or B)")

class Q3Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.q3signature = dspy.ChainOfThought(Q3Signature)

    def forward(self, category: str, document_a: str, document_b: str):
        closest = self.q3signature(CATEGORY=category, DOCUMENT_A=document_a, DOCUMENT_B=document_b).CLOSEST
        
        if "a" in closest.lower():
            closest = "A"
        elif "b" in closest.lower():
            closest = "B"
        else:
            closest = "1000"
        
        return dspy.Prediction(closest = closest)

"""
class Q3SignatureWithExemplars(dspy.Signature):
    #Determine which document is more closely related to the given category
    KEYWORDS = dspy.InputField(desc="Terms that reflect the core theme of the category.")
    EXEMPLAR_DOCUMENTS = dspy.InputField(desc="Sample documents focused on a shared topic related to the category.")
    CATEGORY = dspy.InputField("Represents the common theme among the keywords and exemplar documents.")
    EVALUATION_DOCUMENT_A = dspy.InputField(desc="Evaluation document A")
    EVALUATION_DOCUMENT_B = dspy.InputField(desc="Evaluation document B")
    CLOSEST = dspy.OutputField(desc="Document that is more closely related to the category (A or B)")

class Q3ModuleWithExemplars(dspy.Module):
    def __init__(self):
        super().__init__()
        self.q3signature = dspy.ChainOfThought(Q3SignatureWithExemplars)

    def forward(self, category: str, document_a: str, document_b: str):
        closest = self.q3signature(CATEGORY=category, DOCUMENT_A=document_a, DOCUMENT_B=document_b).CLOSEST
        
        if "a" in closest.lower():
            closest = "A"
        elif "b" in closest.lower():
            closest = "B"
        else:
            closest = "1000"
        
        return dspy.Prediction(closest = closest)
"""

class Q2Signature(dspy.Signature):
    """Determine whether the DOCUMENT fits with the given CATEGORY or not"""
    CATEGORY = dspy.InputField()
    DOCUMENT = dspy.InputField()
    FIT = dspy.OutputField(desc="Whether the DOCUMENT fits with the given CATEGORY or not (YES if it fits, NO if it does not).")
    
class Q2Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.q2signature = dspy.ChainOfThought(Q2Signature)

    def forward(self, category: str, document: str):
        fit = self.q2signature(CATEGORY=category, DOCUMENT=document).FIT
        
        if "yes" in fit.lower():
            fit = "YES"
        elif "no" in fit.lower():
            fit = "NO"
        else:
            print(f"fit: {fit}")
            fit = "10000"
        
        return dspy.Prediction(fit = fit)
    
def get_accuracy(example, pred, trace=None):
    return 1 if example.closest == pred["closest"] else 0

def get_accuracy_fit(example, pred, trace=None):
    return 1 if example.fit == pred["fit"] else 0
    
def optimize_module(data_path, mbd=16, mld=64, ncp=64, mr=5, dev_size=0.25):
                    #mbd=4, mld=16, ncp=16, mr=1, dev_size=0.25):

    dataset = ThetasDataset(data_fpath=data_path, dev_size=dev_size)

    print(f"-- -- Dataset loaded from {data_path}")

    trainset = dataset._train
    devset = dataset._dev
    testset = dataset._test
    
    print(f"-- -- Dataset split into train, dev, and test. Training module...")

    config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                    num_candidate_programs=ncp, max_rounds=mr)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=get_accuracy, **config)

    compiled_pred = teleprompter.compile(Q3Module(), trainset=trainset, valset=devset)

    print(f"-- -- Module compiled. Evaluating on test set...")

    tests = []
    for el in testset:
        output = compiled_pred(el.category, el.document_a, el.document_b)
        tests.append([el.category, el.document_a,  el.document_b, output["closest"], get_accuracy(el, output)])

    df_tests = pd.DataFrame(
        tests, columns=["category", "document_a", "document_b", "closest", "accuracy"])

    print(f"-- -- Test set evaluated. Results:")
    print(df_tests)

    evaluate = Evaluate(
        devset=devset, metric=get_accuracy, num_threads=1, display_progress=True)
    compiled_score = evaluate(compiled_pred)
    uncompiled_score = evaluate(Q3Module())

    print(
        f"## Q3Module Score for uncompiled: {uncompiled_score}")
    print(
        f"## Q3Module Score for compiled: {compiled_score}")
    print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")
    
    return compiled_pred

def optimize_q2_module(data_path, mbd=4, mld=16, ncp=16, mr=1, dev_size=0.25):

    dataset = ThetasDataset(data_fpath=data_path, dev_size=dev_size, mode="q2")

    print(f"-- -- Dataset loaded from {data_path}")

    trainset = dataset._train
    devset = dataset._dev
    testset = dataset._test
    
    print(f"-- -- Dataset split into train, dev, and test. Training module...")

    config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                    num_candidate_programs=ncp, max_rounds=mr)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=get_accuracy_fit, **config)

    compiled_pred = teleprompter.compile(Q2Module(), trainset=trainset, valset=devset)

    print(f"-- -- Module compiled. Evaluating on test set...")

    tests = []
    for el in testset:
        output = compiled_pred(el.category, el.document)
        tests.append([el.category, el.document, output["fit"], get_accuracy_fit(el, output)])

    df_tests = pd.DataFrame(
        tests, columns=["category", "document", "fit", "accuracy"])

    print(f"-- -- Test set evaluated. Results:")
    print(df_tests)

    evaluate = Evaluate(
        devset=devset, metric=get_accuracy_fit, num_threads=1, display_progress=True)
    compiled_score = evaluate(compiled_pred)
    uncompiled_score = evaluate(Q2Module())

    print(
        f"## Q2Module Score for uncompiled: {uncompiled_score}")
    print(
        f"## Q2Module Score for compiled: {compiled_score}")
    print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")
    
    return compiled_pred
        
if __name__ == "__main__":

    """"
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    os.environ["OPENAI_API_KEY"] = api_key
    lm = dspy.LM(model="gpt-4o-mini-2024-07-18")
    dspy.settings.configure(lm=lm)
    """
    

    lm = dspy.LM(
        #"ollama_chat/qwen:32b",
        "ollama_chat/llama3.3:70b",
        api_base="http://kumo01:11434"
    )
    dspy.settings.configure(lm=lm)

    # test the module
    """
    dtset = ThetasDataset(data_fpath="data/files_pilot/user_pairs_tr_data.json")
    trainset = dtset._train
    example = trainset[0]
    pred = Q3Module()(example.category, example.document_a, example.document_b)
    import pdb; pdb.set_trace()
    """
    
    # dspy.inspect_history(n=1)
    
    # Optimize the module
    """
    compiled_pred = optimize_module("data/files_pilot/user_pairs_rank_tr_data.json")
    """
    
    compiled_pred = optimize_q2_module("data/files_pilot/user_fit_tr_data.json")
    compiled_pred.save("data/dspy-saved/q2_llama3.3:70B_26jan.json")
    
    compiled_classifier = compiled_pred
    #compiled_classifier("<<category>>", "<<document>>", "<<document>>")
    compiled_classifier("<<category>>", "<<document>>")
    
    prompt_data = lm.history[-1]
    
    import pdb; pdb.set_trace()