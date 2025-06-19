import argparse
import logging
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from proxann.utils.file_utils import init_logger, split_into_chunks


class Embedder(object):
    def __init__(
        self,
        batch_size: int = 128,
        sbert_model: str = "all-MiniLM-L6-v2",
        aggregate_embeddings: bool = False,
        use_gpu: bool = True,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("config/config.conf")
    ) -> None:

        self._logger = logger if logger else init_logger(config_path, __name__)

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

        self._logger.info(f"Embeddings calculation starts...")
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
                        f"{len(text_chunks)} chunks created. Embeddings calculation starts...")
                else:
                    self._logger.info(
                        f"Chunking was not necessary. Embeddings calculation starts ...")
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
            f"Embeddings extraction finished in {(time.time() - start_time)} seconds")

        return df


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--config_path",
        help="Path to the configuration file.",
        type=str,
        default="config/config.conf",
        required=False
    )
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
        default="all-MiniLM-L6-v2",
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

    # Initialize the logger
    logger = init_logger(args.config_path, Embedder.__name__)

    # Log data loading information
    logger.info(f"Reading data from {args.source_file}...")

    df = pd.read_json(args.source_file, lines=True)
    logger.info(f"Data shape: {df.shape}.")
    logger.info(f"Sample data: {df.head()}.")

    logger.info("Embedding calculation starts...")
    start_time = time.time()

    # Create an instance of the Embedder class
    emb_calculator = Embedder(
        batch_size=args.batch_size,
        sbert_model=args.sbert_model,
        aggregate_embeddings=args.aggregate_embeddings,
        logger=logger
    )

    # Calculate the embeddings
    df_with_embeddings = emb_calculator.calculate_embeddings(
        df,
        col_calculate_on=args.calculate_on
    )

    elapsed_time = time.time() - start_time
    logger.info(
        f"Embedding calculation finished in {elapsed_time} seconds.")

    # Save the dataframe with embeddings
    output_path = pathlib.Path(args.output_file).with_suffix(
        f".{args.sbert_model}.parquet")
    df_with_embeddings.to_parquet(output_path, index=False)
    logger.info(f"Data with embeddings saved to {output_path}.")


if __name__ == "__main__":
    main()
