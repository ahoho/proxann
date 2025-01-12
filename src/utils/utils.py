import configparser
import logging
import pathlib
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def log_or_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Helper function to log or print messages.

    Parameters
    ----------
    message : str
        The message to log or print.
    level : str, optional
        The logging level, by default "info".
    logger : logging.Logger, optional
        The logger to use for logging, by default None.
    """
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
    else:
        print(message)


def init_logger(
    config_file: str,
    name: str = None
) -> logging.Logger:
    """
    Initialize a logger based on the provided configuration.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """

    config = configparser.ConfigParser()
    config.read(config_file)
    name = name if name else config["logger"]["logger_name"]
    log_level = config['logger'].get("log_level", "INFO").upper()
    dir_logger = pathlib.Path(config['logger'].get("dir_logger", "logs"))
    N_log_keep = int(config['logger'].get("N_logs_keep", 5))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create path_logs dir if it does not exist
    dir_logger.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in {dir_logger}")

    # Generate log file name based on the data
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_log_{current_date}.log"
    log_file_path = dir_logger / log_file_name

    # Remove old log files if they exceed the limit
    log_files = sorted(dir_logger.glob("*.log"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    if len(log_files) >= N_log_keep:
        for old_file in log_files[N_log_keep - 1:]:
            old_file.unlink()

    # Create handlers based on config
    if config["logger"].get("file_log", True):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if config["logger"].get("console_log", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


def unpickler(file: str) -> object:
    """
    Unpickle file

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
    """
    Pickle object to file

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
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def file_lines(fname: pathlib.Path) -> int:
    """
    Count number of lines in file

    Parameters
    ----------
    fname: pathlib.Path
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


def get_embeddings_from_str(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Get embeddings from a DataFrame, assuming there is a column named 'embeddings' with the embeddings as strings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the embeddings as strings in a column named 'embeddings'.
    logger : Union[logging.Logger, None], optional
        Logger for logging errors, by default None.

    Returns
    -------
    np.ndarray
        Array of embeddings.

    Raises
    ------
    KeyError
        If the 'embeddings' column is in the df.
    ValueError
        If the 'embeddings' column contains invalid data that cannot be converted to embeddings.
    """

    # Check if 'embeddings' column exists
    if "embeddings" not in df.columns:
        err_msg = "DataFrame does not contain 'embeddings' column."
        log_or_print(err_msg, level="error", logger=logger)
        raise KeyError(err_msg)

    # Extract embeddings
    embeddings = df.embeddings.values.tolist()

    # Check if embeddings are in string format and convert
    try:
        if isinstance(embeddings[0], str):
            embeddings = np.array(
                [np.array(el.split(), dtype=np.float32) for el in embeddings]
            )
        else:
            raise ValueError(
                "Embeddings are not in the expected string format.")
    except Exception as e:
        err_msg = f"Error processing embeddings: {e}"
        log_or_print(err_msg, level="error", logger=logger)
        raise ValueError(err_msg)

    return np.array(embeddings)


def keep_top_k_values(
    matrix: np.ndarray,
    top_k: int = 100,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
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

    Raises
    ------
    ValueError
        If top_k exceeds the number of columns in the matrix.
    """

    if top_k > matrix.shape[1]:
        err_msg = f"top_k ({top_k}) exceeds the number of columns ({
            matrix.shape[1]}) in the matrix."
        log_or_print(err_msg, level="error", logger=logger)
        raise ValueError(err_msg)

    modified_matrix = np.copy(matrix)
    for row in modified_matrix:
        top_k_indices = np.argpartition(row, -top_k)[-top_k:]
        mask = np.zeros_like(row, dtype=bool)
        mask[top_k_indices] = True
        row[~mask] = 0

    return modified_matrix


def load_vocab_from_txt(
    vocab_file: str,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Load vocabulary from a text file.

    Parameters
    ----------
    vocab_file : str
        Path to the vocabulary file.

    Returns
    -------
    dict
        Dictionary mapping words to their indices.
    """
    try:
        vocab_w2id = {}
        with open(vocab_file, 'r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                wd = line.strip()
                vocab_w2id[wd] = i
        return vocab_w2id
    except FileNotFoundError:
        err_msg = f"Vocabulary file not found: {vocab_file}"
        log_or_print(err_msg, level="error", logger=logger)
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    except Exception as e:
        err_msg = f"Error loading vocabulary: {e}"
        log_or_print(err_msg, level="error", logger=logger)
        raise RuntimeError(f"Error loading vocabulary: {e}")


def read_dataframe(
    path_to_data: pathlib.Path,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Read a dataframe from a file.
    Supported file formats: parquet, json, jsonl.

    Parameters
    ----------
    path_to_data : pathlib.Path
        Path to the file containing the data.
    logger : logging.Logger, optional
        Logger for logging messages. Defaults to None.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data.

    Raises
    ------
    ValueError
        If the file format is unsupported.
    RuntimeError
        If an error occurs while reading the data.
    """
    try:
        log_or_print(
            f"Loading data from {path_to_data}...", logger=logger)

        # Determine file format based on the suffix
        if path_to_data.suffix == ".parquet":
            df = pd.read_parquet(path_to_data)
        elif path_to_data.suffix in {".json", ".jsonl"}:
            df = pd.read_json(path_to_data, lines=True)
        else:
            err_msg = f"Unsupported file format: {path_to_data.suffix}"
            log_or_print(err_msg, level="error", logger=logger)
            raise ValueError(err_msg)

        log_or_print(f"Data successfully loaded. Shape: {df.shape}")
        return df

    except Exception as e:
        err_msg = f"An error occurred while reading the data: {e}"
        log_or_print(err_msg, level="error", logger=logger)
        raise RuntimeError(err_msg)
