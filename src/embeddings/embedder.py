import argparse
import logging
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.utils import split_into_chunks


class Embedder(object):
    def __init__(
        self,
        batch_size: int = 128,
        sbert_model: str = "multi-qa-mpnet-base-dot-v1",
        aggregate_embeddings: bool = False,
        use_gpu: bool = True,
        logger: logging.Logger = None
    ) -> None:

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._batch_size_embeddings = batch_size
        self._sbert_model = sbert_model
        self._aggregate_embeddings = aggregate_embeddings
        self._use_gpu = use_gpu
        
    def calculate_embeddings(
        self,
        df: pd.DataFrame,
        col_calculate_on: str,
    ):
        """Calculate embeddings for text columns in a dataframe using SentenceTransformer. The embeddings are saved in a new column named 'embeddings'.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing text columns.
        col_calculate_on : str
            Column name to calculate embeddings on.

        Returns
        -------
        pd.DataFrame
            Dataframe with embeddings added.
        """

        self._logger.info(f"-- -- Embeddings calculation starts...")
        start_time = time.time()

        device = 'cuda' if self._use_gpu and torch.cuda.is_available() else 'cpu'

        model = SentenceTransformer(
            self._sbert_model,
            device=device)
        
        def get_embedding(text):
            """Get embeddings for a text using SentenceTransformer.
            """
            return model.encode(
                text,
                show_progress_bar=True,
                batch_size=self._batch_size_embeddings
            )

        def encode_text(text):
            """Encode text into embeddings using SentenceTransformer. If the text is too long for the model and self._aggregate_embeddings is set to True, it will be split into chunks and the embeddings will be averaged. Otherwise, the embeddings will be calculated only for the part of the text that fits the model's maximum sequence length."""
            
            if self._aggregate_embeddings:
                if len(text) > model.get_max_seq_length():
                    # Split the text into chunks
                    text_chunks = split_into_chunks(
                        text, model.get_max_seq_length())
                    self._logger.info(
                        f"-- -- {len(text_chunks)} chunks created. Embeddings calculation starts...")
                else:
                    self._logger.info(
                        f"-- -- Chunking was not necessary. Embeddings calculation starts ...")
                    text_chunks = [text]
            else:
                text_chunks = [text]

            embeddings = []
            for i, chunk in tqdm(enumerate(text_chunks)):
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
            
            if len(embeddings) > 1:
                embeddings = np.mean(embeddings, axis=0)
            else:
                embeddings = embeddings[0]

            # Convert to string to save space
            embedding_str = ' '.join(str(x) for x in embeddings)
            return embedding_str

        df["embeddings"] = df[col_calculate_on].apply(encode_text)

        self._logger.info(
            f"-- -- Embeddings extraction finished in {(time.time() - start_time)} seconds")

        return df

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--source_file",
        help="Path to the source file containing the data to be processed. The file should be in JSON format.",
        type=str,
        default="data/train.metadata.jsonl",
        required=False
    )
    argparser.add_argument(
        "--output_file",
        help="Path to the output file where the processed data will be saved. The file will be saved in PARQUET format.",
        type=str,
        default="data/train.metadata.parquet",
        required=False
    )
    argparser.add_argument(
        "--batch_size",
        help="Batch size for SentenceTransformer.",
        type=int,
        default=128,
        required=False
    )
    argparser.add_argument(
        "--sbert_model",
        help="SentenceTransformer model to use.",
        type=str,
        default="multi-qa-mpnet-base-dot-v1",# all-MiniLM-L6-v2 is the default of bertopic
        required=False
    )
    argparser.add_argument(
        "--aggregate_embeddings",
        help="Whether to aggregate embeddings for long texts.",
        type=bool,
        default=False,
        required=False
    )
    argparser.add_argument(
        "--calculate_on",
        help="Column to calculate embeddings on.",
        type=str,
        default="text",
        required=False
    )

    args = argparser.parse_args()

    # Read data
    df = pd.read_json(args.source_file, lines=True)

    print(f"-- -- Data read from {args.source_file}.")
    print(f"-- -- Data shape: {df.shape}.")
    print(f"-- -- Sample data: {df.head()}.")

    print(f"-- --  Embedding calculation starts... ")
    start_time = time.time()

    # Create an instance of the class
    emb_calculator = Embedder(
        batch_size=args.batch_size,
        sbert_model=args.sbert_model,
        aggregate_embeddings=False,
    )

    # Calculate the embeddings
    df_with_embeddings = emb_calculator.calculate_embeddings(
        df,
        col_calculate_on=args.calculate_on)

    end_time = time.time() - start_time
    print(f"-- -- Embedding calculation finished in {end_time}")

    # Save the dataframe with embeddings
    path_save = pathlib.Path(args.output_file).parent / (pathlib.Path(args.output_file).name + f".{args.sbert_model}.parquet")
    df_with_embeddings.to_parquet(path_save, index=False)

    print(f"-- -- Data with embeddings saved to {args.output_file}.")


if __name__ == "__main__":
    main()
