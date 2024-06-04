import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union

def unpickler(file: str) -> object:
    """Unpickle file

    Parameters
    ----------
    file : str
        The path to the file to unpickle.

    Returns
    -------
    object
        The unpickled object.
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob: object) -> int:
    """Pickle object to file

    Parameters
    ----------
    file : str
        The path to the file where the object will be pickled.
    ob : object
        The object to pickle.

    Returns
    -------
    int
        0 if the operation is successful.
    """
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return 0

def split_into_chunks(text: str, max_length: int) -> List[str]:
    """
    Split a text into chunks of a specified maximum length.

    Parameters
    ----------
    text : str
        The text to be split into chunks.
    max_length : int
        The maximum length of each chunk.

    Returns
    -------
    list of str
        A list containing the text split into chunks.
    """
    if len(text) > max_length:
        texts_splits = [text[i:i + max_length]
                        for i in range(0, len(text), max_length)]
    else:
        texts_splits = [text]
    return texts_splits

def init_logger(name: str, path_logs: Path) -> logging.Logger:
    """
    Initialize a logger with a specified name and log file path.

    Parameters
    ----------
    name : str
        The name of the logger.
    path_logs : Path
        The directory path where the log file will be stored.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create path_logs dir if it does not exist
    path_logs.mkdir(parents=True, exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(path_logs / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)

    # Add the handlers to the logger if they are not already added
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def file_lines(fname: Path) -> int:
    """
    Count number of lines in file

    Parameters
    ----------
    fname: Path
        The file whose number of lines is calculated.

    Returns
    -------
    int
        Number of lines in the file.
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_embeddings_from_str(df: pd.DataFrame, logger: Union[logging.Logger, None] = None) -> np.ndarray:
    """
    Get embeddings from a DataFrame, assuming there is a column named 'embeddings' with the embeddings as strings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the embeddings as strings in a column named 'embeddings'
    logger : Union[logging.Logger, None], optional
        Logger for logging errors, by default None

    Returns
    -------
    np.ndarray
        Array of embeddings
    """

    if "embeddings" not in df.columns:
        if logger:
            logger.error(
                f"-- -- DataFrame does not contain embeddings column"
            )
        else:
            print(
                f"-- -- DataFrame does not contain embeddings column"
            )
        
    embeddings = df.embeddings.values.tolist()
    if isinstance(embeddings[0], str):
        embeddings = np.array(
            [np.array(el.split(), dtype=np.float32) for el in embeddings])

    return np.array(embeddings)

def keep_top_k_values(matrix: np.ndarray, top_k: int = 100) -> np.ndarray:
    """
    For each row in the matrix, keep the top_k largest values and set the rest to zero.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of dimensions K x V.
    top_k : int, optional
        Number of largest values to keep in each row, by default 100.

    Returns
    -------
    np.ndarray
        The modified matrix with only the top_k values kept in each row.
    """
    
    modified_matrix = np.copy(matrix)
    for row in modified_matrix:
        top_k_indices = np.argpartition(row, -top_k)[-top_k:]        
        mask = np.zeros_like(row, dtype=bool)
        mask[top_k_indices] = True
        row[~mask] = 0
        
    return modified_matrix
