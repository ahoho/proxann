<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="./figures/repo/Proxann8.png" alt="Logo" width="90" height="90" style="vertical-align: middle; margin-right: 10px;">
  <span style="font-size: 2em;">THETA-EVALUATION</span>
</p>
<!-- markdownlint-enable MD033 -->
<!-- markdownlint-disable MD041 -->

This repository contains the code and data for reproducing experiments from our paper, *ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering.*

## ✨ Features

1. **User Study Data Generation**:
   - Use the `src.user_study_data_collector` module to generate JSON files containing topic model information for conducting user studies.

2. **Proxy-Based Evaluation**:
   - Perform LLM proxy annotations using the `src.proxann` module.

3. **Topic Model Training**:
   - Train topic and clustering models (currently, LDA-Mallet, LDA-Tomotopy, and BERTopic) under a unified structure using the `src.train` module.

## 🛠️ Installation

[Add some details on installation using poetry]

> **IMPORTANT TO RUN LDA-Mallet**
>
> Download the [latest release of Mallet](https://github.com/mimno/Mallet/releases) and place it in the `src/train` directory. This can be done using the script `bash_scripts/wget_mallet.sh`.

## 🚀 Usage

[complete when "metric" module is ready]

## 🔄 Reproducibility

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
python3 get_user_study_data.py --config <path_user_study_config_file>
```

### Human Annotations

User responses were collected through Prolific using the `src.annotations.annotation_server` server. The collected responses are XXX [confirm whether they are uploaded to Git].
Here’s an improved version of your section with enhanced clarity, grammar corrections, and better flow:

### LLM Annotations

To obtain LLM annotations, run the `proxann.sh` script with the following parameters:

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