<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->
<p align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="./figures/repo/Proxann8.png" alt="Logo" width="150" height="150" style="display: inline-block;">
</p>
<!-- markdownlint-enable MD033 -->
<!-- markdownlint-disable MD041 -->

This repository contains the code and data for reproducing experiments from our paper, *ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering.*

## Features

1. **User Study Data Generation**:
   - Use the `src.user_study_data_collector` module to generate the JSON files containing the required topic model information to carry out the evaluation (or user study).
2. **Proxy-Based Evaluation**:
   - Perform LLM proxy evaluations using the `src.proxann` module.

3. **Topic Model Training**:
   - Train topic and clustering models (currently, LDA-Mallet, LDA-Tomotopy, and BERTopic) under a unified structure using the `src.train` module.

## Installation

We recommend **Poetry** for installing the necessary dependencies, but an environment followed by the installation of the [requirements file](requirements.txt) can also be used.

### Steps for deployment with Poetry

1. Install Poetry by following the official guide: [Poetry Installation Guide](https://python-poetry.org/docs/#installing-with-the-official-installer)

2. Verify that Poetry is using Python 3.11 (preferably version 3.11.11):
   ```bash
   poetry run python --version
   ```

3. Install project dependencies:
   ```bash
   poetry install
   ```

### LLM configuration
#### GPT models
You must configure an ``.env`` file located in the root directory with the following format:
```bash
OPENAI_API_KEY=[your_open_ai_api_key]
```
You can also modify the path to the ``.env`` file in the [configuration file](config/config.yaml).

#### Open-source models
We rely on [Ollama models](https://ollama.com/) for evaluating with open-source large language models. You must have the model running and specify the endpoint where it is deployed in the [configuration file](config/config.yaml).

### Important: Training with LDA-Mallet
Download the [latest release of Mallet](https://github.com/mimno/Mallet/releases) and place it in the `src/train` directory. You can use the script `bash_scripts/wget_mallet.sh` to automate this process.

## Usage

You can use **ProxAnn** as a proxy for human annotators to evaluate the quality of topic models. To use it:

1. Initialize a [ProxAnn](src/proxann/proxann.py) object.
    ```python
    from src.proxann.proxann import ProxAnn
    proxann = ProxAnn()
    ```
2. Generate a user-provided JSON file using the user study configuration (``path_user_study_config_file``). You can find example configurations [here](config/user_study).
    ```python
    status, tm_model_data_path = proxann.generate_user_provided_json(path_user_study_config_file)
    ```
    - If ``status == 0``, the JSON file was generated successfully.
    - Otherwise, an error occurred, and execution should be stopped.

3. Run the evaluation metrics using the ``run_metric()`` method, specifying the generated JSON file and the LLM model(s) to use:
    ```python
    proxann.run_metric(
        tm_model_data_path.as_posix(),
        llm_models=["qwen:32b"]
    )
    ```
The script [``proxann_eval.py``](proxann_eval.py) contains a simple demonstration of this.

### Deploying ProxAnn as a Web Service

To run ProxAnn as a web service locally, execute the following command:

```bash
python3 -m src.metric_mode.back
```

This will start a web server that exposes ProxAnn's functionality via a REST API.

Alternatively, you can use our hosted instance of the ProxAnn web service here: [https://proxann.uc3m.es/](https://proxann.uc3m.es/)


## Reproducibility

### Data

We use the Wiki and Bills preprocessed datasets from [Hoyle et al. 2022](https://aclanthology.org/2022.findings-emnlp.390/) in their 15,000-term vocabulary form, available in the [original repository](https://github.com/ahoho/topics).

### Models

We reuse the 50-topics LDA-Mallet and CTM topic models from [Hoyle et al. 2022](https://aclanthology.org/2022.findings-emnlp.390/) and train a BERTopic model for each dataset using the same experimental setup with default hyperparameters, implemented in the `src.train.tm_trainer.BERTopicTrainer` class. These trained models are available upon request.

### User Study Configuration

We randomly sample 8 of the 50 topics from the Wiki and Bills datasets for each of the three models. For each sampled topic, the corresponding topics from the remaining models are selected based on the smallest word-mover's distance, as implemented in the `src.user_study_data_collector.topics_docs_selection.topic_selector` class.

Using the [user study configuration files](config/user_study), JSON files (one per dataset) can be generated to set up the user and ProxAnn studies. These configuration files must adhere to the following structure:

- **Model-Specific Settings:**
  For each model being evaluated, specify the paths based on its training origin:
  - If the model was **not trained using this code** (`trained_with_thetas_eval=False`), provide:
    - `thetas_path`: Path to the document-topic distribution.
    - `betas_path`: Path to the word-topic distribution.
    - `vocabulary_path`: Path to the vocabulary file.
    - `corpus_path`: Path to the corpus file.
  - If the model **was trained using this code** (`trained_with_thetas_eval=True`), provide:
    - `model_path`: Path to the model file.
    - `corpus_path`: Path to the corpus file.
  - Additionally, `remove_topic_ids` can contain numbers (separated by commas) representing topics that should not be considered for matching.

The generated JSON files are available [here](data/json_out) and can be created using the following command:

```bash
python3 get_user_study_data.py --user_study_config <path_user_study_config_file>
```

### Human Annotations

User responses were collected through Prolific using the `src.annotations.annotation_server` server. The collected responses are XXX [update this depending on where the files are saved].

### LLM Annotations

To obtain LLM annotations, run the `proxann_user_study.py` (or its bash version `bash_scripts/run_proxann.sh`) script with the following parameters:

- **`--model_type "$MODEL_TYPE"`**  
  Specifies the language model(s) to be used for generating annotations. Both open-source and closed-source models are supported. Refer to `config/config.yaml` for the currently available models. New models can be added as needed.

- **`--prompt_mode "$PROMPT_MODE"`**  
  Defines the evaluation steps to perform. Options include:
  - `q1_then_q2_dspy`: Step 1 – Category Identification, followed by Step 2 – Relevance Judgment.  
  - `q1_then_q3_dspy`: Step 1 – Category Identification, followed by Step 3 – Representativeness Ranking.  
  Multiple modes can be specified simultaneously, separated by commas.

- **`--removal_condition "$REMOVAL_CONDITION"`**  
  Determines the criteria for disqualifying responses:  
  - `loose`: Disqualifies responses with one or more failures.  
  - `strict`: Disqualifies responses only if all conditions fail.

- **`--path_save_results "$SAVE_PATH"`**  
  Specifies the directory path where the generated annotations will be saved.

- **`--tm_model_data_path "$TM_MODEL_DATA_PATH"`**  
  Path to the user study JSON files generated during the setup phase.

- **`--response_csv "$RESPONSE_CSV"`**  
  Path to the CSV file containing human annotations collected through Qualtrics.

- **`--dataset_key "$DATASET_KEY"`**  
  Identifies the dataset to be annotated (e.g., `Wiki`, `Bills`).