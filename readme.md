# THETA-EVALUATION

## Modules

### Embedder

Calculates embeddings using `SentenceTransformers`, with or without aggregation, and saves them as strings in a new dataframe column.

### TopicJsonFormatter

Generates JSON output for topic models, including:

- **Representative Documents**
- **Top Words for Each Topic**
- **Evaluation Documents and Their Probabilities**

### DocSelector

Implements methods to select representative documents for each topic:

- **thetas**: Selects the ntop-documents with the highest thetas (doc-topic distrib).
- **thetas_sample**: The ntop-documents are selected based on probabilistic sampling. The thetas matrix is normalized such that the columns sum to 1. For each topic, documents are sampled according to their probabilities in the normalized thetas matrix.
- **thetas_thr**: The ntop-documents are selected based on a threshold. The thetas matrix is filtered such that only values within the specified threshold range are kept.
- **sall**: Top docs are selected based on the largest Bhattacharya coefficient between their normalized BoW and the betas.
- **spart**: Top docs are chosen by identifying those with the largest Bhattacharya coefficient between the BoW of the document, specific to the words generated for the topic, and the topic's betas.
- **s3**: For each topic, top docs are chosen by keeping those with the largest sum of the weights that such a topic assigns to each word in the document.
- **elbow**: Selects top docs by identifying the elbow point for each topic. Documents are chosen based on the probabilities in the thetas matrix after filtering out values below the elbow point.

### TopicSelector

Implements methods to find optimal pairings between topics from different models.

### TMTrainer

Provides different topic modeling wrappers: ``LDA-Mallet``, ``LDA-Tomotopy``, and ``BERTopic``.

> **IMPORTANT TO RUN LDA-Mallet**
>
> Download the [latest release of Mallet](https://github.com/mimno/Mallet/releases) and place it in the `src/train` directory. This can be done using the script `bash_scripts/wget_mallet.sh`.

## Running the Modules

All modules can be run via the `main.py` file.

**Example usage:**

```bash
python main.py generate_embeddings --source_file=<path> --output_file=<path> --batch_size=128
python main.py train_tm --corpus_file=<path> --model_path=<path> --trainer_type=MalletLda --num_topics=50
python main.py get_top_docs --method=thetas --thetas_path=<path> --ntop=5
python main.py jsonfy --method=thetas --thetas_path=<path> --ntop=5
```

The ``bash_scripts`` folder contains one bash script per functionality on how to run them.

## Pilot

## Get Data for Pilot

Configure the `config/config_pilot.conf` file and execute the script `bash_scripts/jsonfy_pilot.sh`. The script dumps a JSON, which is also saved at `data/json_out/config_pilot.json`.