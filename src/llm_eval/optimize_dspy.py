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

from utils import extend_to_full_sentence

class ThetasDataset(Dataset):

    def __init__(
        self,
        data_fpath: str,
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        seed: Optional[int] = 11235,
        inputs = ["category", "document_a", "document_b"],
        outputs = ["closest"],
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
        
        # keep first category from each user
        train_data["category"] = train_data["categories"].apply(lambda x: x[0])
        train_data["closest"] = train_data.apply(lambda x: "A" if x["users_fit_winner"] == x["doc_id1"] else "B", axis=1)
        
        # truncate the documents to num_words_truncate
        train_data["document_a"] = train_data["doc1"].apply(lambda x: extend_to_full_sentence(x,num_words_truncate))
        train_data["document_b"] =  train_data["doc2"].apply(lambda x: extend_to_full_sentence(x,num_words_truncate))
        
        train_data = train_data[(inputs + outputs)]
        
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
        return dspy.Prediction(closest = closest)
    
def get_accuracy(example, pred, trace=None):
    return 1 if example.closest == pred["closest"] else 0
    
def optimize_module(data_path, mbd=4, mld=16, ncp=16, mr=1, dev_size=0.25):

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
        
if __name__ == "__main__":
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    lm = dspy.LM(model="gpt-4o-mini-2024-07-18")
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
    compiled_pred = optimize_module("data/files_pilot/user_pairs_tr_data.json")
    compiled_pred.save("data/dspy-saved/compiled-q3-gpt-4o-mini-2024-07-18.json")
    
    compiled_classifier = compiled_pred
    compiled_classifier("<<category>>", "<<document_a>>", "<<document_b>>")
    
    prompt_data = lm.history[-1]
    
    import pdb; pdb.set_trace()