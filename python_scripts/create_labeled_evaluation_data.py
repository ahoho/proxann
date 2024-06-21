#%%
import sys
sys.path.append("..")
from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_extraction.text import CountVectorizer

from bertopic.vectorizers import ClassTfidfTransformer

from src.utils.utils import get_embeddings_from_str

#%%
if __name__ == "__main__":
    # assumes that python `main.py` has been run already
    parser = argparse.ArgumentParser()
    # parser.add_argument("--evaluation_json", type=str, default="evaluation.json")
    parser.add_argument("--output_dir")
    parser.add_argument("--parquet_data_file", type=str, default="data")
    parser.add_argument("--use_medoid", action="store_true", help="Use medoid instead of centroid")
    parser.add_argument("--group_vars", type=str, nargs="+", default=["supercategory", "category", "subcategory"])
    args = parser.parse_args()

    #%%
    # eval_data = json.loads(Path(args.evaluation_json).read_text())
    document_data = pd.read_parquet(args.parquet_data_file)
    # assert the index is a range index
    assert document_data.index.equals(pd.RangeIndex(len(document_data)))
    embeds = get_embeddings_from_str(document_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    #%% create approximations of theta and beta parameters for each grouping variable
    for group_var in args.group_vars:
        # initialize pseudo theta and per-group vocabulary counts
        n_groups = document_data[group_var].nunique()
        pseudo_theta = np.zeros((len(document_data), n_groups))
        group_docs = []

        # first, construct the pseudo-theta
        # for each group, calculate the central embedding and the similarity of each document to the central embedding
        print(f"Building pseudo-theta for {group_var}")
        for i, (group, group_data) in enumerate(document_data.groupby(group_var)):
            group_embeds = embeds[group_data.index]
            if args.use_medoid:
                group_dists = cosine_distances(group_embeds)
                medoid_idx = np.argmin(group_dists.sum(axis=0))
                central_emb = group_embeds[medoid_idx]
            else:
                central_emb = group_embeds.mean(axis=0)
            
            sims = cosine_similarity(embeds, central_emb[None, :]).squeeze()
            pseudo_theta[:, i] = sims
            # asserts that if we take the argmax for a document, it is the same as the group it belongs to
            pseudo_theta[group_data.index] += 1.
            
            # for building c-tf-idf
            group_docs.append(" ".join(group_data["tokenized_text"].tolist()))

        # then, make the grouped c-tf-idf for the pseudo-beta
        # create vocabulary with a count vectorizer with no processing at all
        print(f"Building pseudo-beta for {group_var}")
        cv = CountVectorizer(
            analyzer=lambda x: x.split(),
            lowercase=False,
            stop_words=None,
        )
        group_tm = cv.fit_transform(group_docs)
        
        # transform the group term matrix into a c-tf-idf matrix
        ctfidf = ClassTfidfTransformer()
        ctfidf.fit(group_tm)
        pseudo_beta = ctfidf.transform(group_tm)
        group_output_dir = (output_dir / f"{group_var}-{n_groups}")
        group_output_dir.mkdir(parents=True, exist_ok=True)
        np.save(group_output_dir / "theta.npy", pseudo_theta)
        np.save(group_output_dir / "beta.npy", pseudo_beta)
        with open(group_output_dir / "vocab.json", "w") as outfile:
            json.dump(cv.vocabulary_, outfile)
# %%
