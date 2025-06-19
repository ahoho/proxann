<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->
<p align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="./figures/repo/Proxann8.png" alt="Logo" width="150" height="150" style="display: inline-block;">
</p>
<!-- markdownlint-enable MD033 -->
<!-- markdownlint-disable MD041 -->

This repository contains the code and data for reproducing experiments from our paper, *ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering.*

- [**Features**](#features)
- [**Installation**](#installation)
  - [Steps for deployment with uv](#steps-for-deployment-with-uv)
- [**Configuration**](#configuration)
  - [1. LLMs](#1-llms)
    - [GPT Models (OpenAI)](#gpt-models-openai)
    - [Open-Source Models (via vLLM)](#open-source-models-via-vllm)
  - [2. Topics Models](#2-topics-models)
- [**Getting Started**](#getting-started)
  - [**Choosing the Right LLM for ProxAnn Metrics**](#choosing-the-right-llm-for-proxann-metrics)
    - [**Alternative Annotator Test**](#alternative-annotator-test)
    - [**Relationship between automated and human topic rankings**](#relationship-between-automated-and-human-topic-rankings)
  - [**Preparing Topic Models for ProxAnn**](#preparing-topic-models-for-proxann)
  - [**Generating User Study Input**](#generating-user-study-input)
    - [**Configuration Files**](#configuration-files)
    - [**Generating JSON Files**](#generating-json-files)
  - [**Evaluating Topic Models with ProxAnn Metrics**](#evaluating-topic-models-with-proxann-metrics)
    - [1. Initialize the ProxAnn object](#1-initialize-the-proxann-object)
    - [2. Generate the user study JSON](#2-generate-the-user-study-json)
    - [3. Run the ProxAnn evaluation](#3-run-the-proxann-evaluation)
    - [Running ProxAnn via Web Service](#running-proxann-via-web-service)
  - [**Running LLMs Independently (Prompter)**](#running-llms-independently-prompter)
    - [1. Initialize the `Prompter`](#1-initialize-the-prompter)
    - [2. Make a Prompt Call](#2-make-a-prompt-call)
  - [**Human Annotations**](#human-annotations)
  - [**Evaluating New LLMs with Human and Topic Model Data**](#evaluating-new-llms-with-human-and-topic-model-data)
  - [**Reproducing Results from the Paper**](#reproducing-results-from-the-paper)
    - [Data](#data)


## **Features**

1. **User Study Data Generation**:
   - Use the `src.user_study_data_collector` module to generate the JSON files containing the required topic model information to carry out the evaluation (or user study).
2. **Proxy-Based Evaluation**:
   - Perform LLM proxy evaluations using the `src.proxann` module.
3. **Topic Model Training**:
   - Train topic and clustering models (currently, LDA-Mallet, LDA-Tomotopy, and BERTopic) under a unified structure using the `src.train` module.

## **Installation**

We recommend **uv** for installing the necessary dependencies.

### Steps for deployment with uv

1. Install uv by following the [official guide](https://docs.astral.sh/uv/getting-started/installation/)

2. Create a local environment (it will use the python version specified in pyproject.toml)
  ```bash
  uv venv
  ```

3. Install dependencies
  ```bash
  uv pip install -e .
  ```

3. Run scripts in this repository with either `uv run <bash script>.sh` or `uv run python <python script>.py`. You can also first run `source .venv/bin/activate` to avoid the need for `uv run`.

## **Configuration**

### 1. LLMs

#### GPT Models (OpenAI)

To use GPT models via the OpenAI API, create a `.env` file in the root directory with the following content:
```bash
OPENAI_API_KEY=[your_open_ai_api_key]
```
You can also modify the path to the ``.env`` file in the [configuration file](config/config.yaml).

#### Open-Source Models (via vLLM)
We rely on [vLLM models](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai) for evaluating with open-source large language models. You must have the model running and specify the endpoint where it is deployed in the [configuration file](config/config.yaml).

### 2. Topics Models

The `src.train` module supports multiple topic modeling backends. No extra setup is required for most of them.

**Only if you're using LDA-Mallet**, follow these steps:

1. Download the [latest release of Mallet](https://github.com/mimno/Mallet/releases).
2. Place the contents in the `src/train` directory.
3. Optionally, you can use the provided script to automate the download:
```bash
bash bash_scripts/wget_mallet.sh
```

## **Getting Started**

This section will guide you through the process of setting up and using ProxAnn, from choosing the right LLM to running your first metric.

### **Choosing the Right LLM for ProxAnn Metrics**

ProxAnn’s performance varies depending on the language model used. This section summarizes how different LLMs perform across tasks and datasets, helping you balance accuracy with computational cost.

> ⚠️ **Recommendation:** For best overall alignment with human judgments, **GPT-4o** and **Qwen 2.5–72B** perform the strongest across both Fit and Rank steps.  
> **Qwen 1.5–32B** is a solid cost-effective alternative.  
> Avoid **Llama 3.1–8B**, which consistently underperforms.

#### **Alternative Annotator Test**

This test estimates how often ProxAnn (with a given LLM) performs *as well as or better than* a random human annotator.
Metrics are **advantage probabilities**. Asterisks (`*`) and daggers (`†`) mark statistical significance: `*` indicates the LLM outperforms a random human annotator (p < 0.05, t-test); `†` shows significance under a Wilcoxon signed-rank test.

<div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">

  <!-- Wiki Table -->
  <div>
    <h4 style="text-align: center;">Wiki</h4>
    <table border="1" cellpadding="6" cellspacing="0">
      <thead>
        <tr>
          <th>Model</th>
          <th>Doc ρ (Fit)</th>
          <th>Doc ρ (Rank)</th>
          <th>Topic ρ (Fit)</th>
          <th>Topic ρ (Rank)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>GPT-4o</td><td>0.56<sup>*†</sup></td><td>0.68<sup>*†</sup></td><td>0.66<sup>†</sup></td><td>0.55<sup>†</sup></td></tr>
        <tr><td>Llama 3.1 8B</td><td>0.22</td><td>0.36</td><td>0.05</td><td>0.11</td></tr>
        <tr><td>Llama 3.1 70B</td><td>0.57<sup>*†</sup></td><td>0.67<sup>*†</sup></td><td>0.58<sup>†</sup></td><td>0.50<sup>†</sup></td></tr>
        <tr><td>Qwen 1.5 8B</td><td>0.56<sup>*†</sup></td><td>0.58<sup>†</sup></td><td>0.46</td><td>0.39</td></tr>
        <tr><td>Qwen 1.5 32B</td><td>0.55<sup>*†</sup></td><td>0.63<sup>†</sup></td><td>0.47</td><td>0.42</td></tr>
        <tr><td>Qwen 2.5 72B</td><td>0.52<sup>†</sup></td><td>0.68<sup>*†</sup></td><td>0.66<sup>†</sup></td><td>0.46</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Bills Table -->
  <div>
    <h4 style="text-align: center;">Bills</h4>
    <table border="1" cellpadding="6" cellspacing="0">
      <thead>
        <tr>
          <th>Model</th>
          <th>Doc ρ (Fit)</th>
          <th>Doc ρ (Rank)</th>
          <th>Topic ρ (Fit)</th>
          <th>Topic ρ (Rank)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>GPT-4o</td><td>0.65<sup>*†</sup></td><td>0.71<sup>*†</sup></td><td>0.77<sup>*†</sup></td><td>0.75<sup>*†</sup></td></tr>
        <tr><td>Llama 3.1 8B</td><td>0.30</td><td>0.53<sup>†</sup></td><td>0.14</td><td>0.44</td></tr>
        <tr><td>Llama 3.1 70B</td><td>0.66<sup>*†</sup></td><td>0.67<sup>*†</sup></td><td>0.70<sup>*†</sup></td><td>0.60<sup>†</sup></td></tr>
        <tr><td>Qwen 1.5 8B</td><td>0.66<sup>*†</sup></td><td>0.57<sup>†</sup></td><td>0.80<sup>*†</sup></td><td>0.43</td></tr>
        <tr><td>Qwen 1.5 32B</td><td>0.67<sup>*†</sup></td><td>0.68<sup>*†</sup></td><td>0.74<sup>*†</sup></td><td>0.70<sup>*†</sup></td></tr>
        <tr><td>Qwen 2.5 72B</td><td>0.61<sup>*†</sup></td><td>0.71<sup>*†</sup></td><td>0.78<sup>*†</sup></td><td>0.65<sup>†</sup></td></tr>
      </tbody>
    </table>
  </div>

</div>

#### **Relationship between automated and human topic rankings**
The plot and table below show how well ProxAnn’s topic rankings align with human judgments, using Kendall’s τ as the correlation metric. The *Human* row reflects inter-annotator agreement, and NPMI provides a traditional baseline.

<div align="center">
  <img src="figures/human_llm_comparison_barplot.png" alt="Human vs LLM correlation barplot" width="1000">
</div>

<br>

<div align="center">
  <table border="1" cellpadding="6" cellspacing="0">
    <thead>
      <tr>
        <th>Metric / Model</th>
        <th>Wiki (Fit)</th>
        <th>Bills (Fit)</th>
        <th>Wiki (Rank)</th>
        <th>Bills (Rank)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>NPMI</td><td>-0.15 (0.14)</td><td>0.01 (0.10)</td><td>-0.18 (0.10)</td><td>-0.02 (0.12)</td></tr>
      <tr><td><strong>GPT-4o</strong></td><td>0.22 (0.13)</td><td>0.31 (0.13)</td><td>0.27 (0.14)</td><td>0.29 (0.11)</td></tr>
      <tr><td><strong>Llama 3.1 8B</strong></td><td>0.19 (0.18)</td><td>0.16 (0.18)</td><td>-0.35 (0.14)</td><td>0.15 (0.14)</td></tr>
      <tr><td><strong>Qwen 1.5 8B</strong></td><td>0.35 (0.16)</td><td>0.12 (0.16)</td><td>0.33 (0.16)</td><td>0.28 (0.13)</td></tr>
      <tr><td><strong>Qwen 1.5 32B</strong></td><td>0.20 (0.18)</td><td><strong>0.34 (0.11)</strong></td><td><strong>0.51 (0.11)</strong></td><td><strong>0.30 (0.13)</strong></td></tr>
      <tr><td><strong>Llama 3.1 70B</strong></td><td>0.41 (0.14)</td><td>0.26 (0.15)</td><td>0.36 (0.13)</td><td>0.19 (0.13)</td></tr>
      <tr><td><strong>Qwen 2.5 72B</strong></td><td><strong>0.48 (0.13)</strong></td><td>0.22 (0.17)</td><td>0.36 (0.12)</td><td>0.21 (0.15)</td></tr>
      <tr><td><em>Human (HTM)</em></td><td>0.41 (0.09)</td><td>0.09 (0.14)</td><td>0.34 (0.09)</td><td>0.18 (0.12)</td></tr>
    </tbody>
  </table>
</div>

---

### **Preparing Topic Models for ProxAnn**

ProxAnn expects output from *traditional topic models*, where each document is represented by a topic distribution ($\theta_d$) and each topic by a word distribution ($\beta_k$). Outputs from *document clustering* can also be used by mapping cluster assignments to $\theta_d$ and generating topic labels (approximating $\beta_k$) via word selection or language model summaries.

To be used with ProxAnn, models must be saved as NumPy arrays (`.npy` or `.npz`), along with:

* A JSON file containing the model vocabulary (i.e., the words indexing the columns in $\beta_k$).
* A plain-text corpus file (one document per line).

Alternatively, you can train topic models directly using ProxAnn's training module. In that case, only the corpus is required. See [`bash_scripts/train_models.sh`](bash_scripts/train_models.sh) for an example of how to invoke [`src/train/tm_trainer.py`](src/train/tm_trainer.py).

---

### **Generating User Study Input**

ProxAnn creates a JSON file that serves as input for both human and LLM-based evaluations. This file contains:

* Top words for each topic (`topic_words`)
* Representative documents (`exemplar_docs`) using various selection methods (`thetas`, `thetas_sample`, `sall`, etc.)
* Evaluation documents with topic assignment probabilities (`eval_docs`)
* A distractor document for each topic

**Example structure:**

```json
{
  "<topic_id>": {
    "topic_words": ["word1", "word2", "word3"],
    "exemplar_docs": [
      {"doc_id": 1, "text": "...", "prob": 0.9},
      {"doc_id": 2, "text": "...", "prob": 0.8}
    ],
    "eval_docs": [
      {"doc_id": 3, "text": "...", "prob": 0.9, "assigned_to_k": 1},
      {"doc_id": 4, "text": "...", "prob": 0.8, "assigned_to_k": 1}
    ],
    "distractor_doc": {"doc_id": 100, "text": "..."}
  }
}
```

---

#### **Configuration Files**

To generate the above JSON files, you’ll need a YAML config file like those in [`config/user_study`](config/user_study). Each config should specify how to load model outputs depending on how the model was trained:

* If the model was **not trained with ProxAnn** (`trained_with_thetas_eval=False`), provide:

  * `thetas_path`: Document-topic matrix (docs × topics)
  * `betas_path`: Topic-word matrix (topics × vocab size)
  * `vocabulary_path`: Vocabulary file
  * `corpus_path`: Original documents (one per line)

* If the model **was trained using ProxAnn** (`trained_with_thetas_eval=True`), provide:

  * `model_path`: Path to the trained model
  * `corpus_path`: As above

* You can also specify `remove_topic_ids` to exclude topics from evaluation.

---

#### **Generating JSON Files**

To create the user study data files, run:

```python
python3 get_user_study_data.py --user_study_config <path_to_config_file>
```

You can see examples of generated JSONs in [`data/json_out`](data/json_out).

---

### **Evaluating Topic Models with ProxAnn Metrics**

To evaluate your topic model using ProxAnn:

#### 1. Initialize the ProxAnn object

```python
from src.proxann.proxann import ProxAnn
proxann = ProxAnn()
```

#### 2. Generate the user study JSON

Use a user study configuration file (see examples in [`config/user_study`](config/user_study)) to produce the input JSON for evaluation:

```python
status, tm_model_data_path = proxann.generate_user_provided_json(path_user_study_config_file)
```

* If `status == 0`, the JSON was created successfully.
* Otherwise, an error occurred, and evaluation should be halted.

#### 3. Run the ProxAnn evaluation

```python
proxann.run_metric(
    tm_model_data_path.as_posix(),
    llm_models=["Qwen/Qwen3-8B"]
)
```

* `llm_models` is a list of LLMs to use for evaluation.
* These must be pre-defined in your [`config/config.yaml`](config/config.yaml), under the deployment section you're using (e.g., `vllm`, `openai`, etc.).

Example `config.yaml` snippet for a VLLM setup:

```yaml
llm:
  vllm:
    available_models:
      "Qwen/Qwen3-8B": ...
    host: http://localhost:8000/v1
```

See [`proxann_eval.py`](proxann_eval.py) for a minimal runnable example.

#### Running ProxAnn via Web Service

You can also run ProxAnn as a REST API server:

```bash
python3 -m src.metric_mode.back
```

This launches a local web server that exposes ProxAnn’s evaluation pipeline via HTTP endpoints.

Alternatively, use the hosted instance at:
👉 [https://proxann.uc3m.es/](https://proxann.uc3m.es/)


---

### **Running LLMs Independently (Prompter)**

ProxAnn uses a unified wrapper class, [`Prompter`](src/proxann/prompter.py), to standardize API calls across different LLM backends. It currently supports **OpenAI**, **VLLM**, and **Ollama**.

> ⚠️ Note: Only **OpenAI** and **VLLM** support logprobs, which are required for ProxAnn’s evaluation. Ollama is currently not compatible for this reason.

The `Prompter` class includes a **caching mechanism** that ensures repeated prompts return the same result without reissuing an API call, improving speed and efficiency during evaluation.

#### 1. Initialize the `Prompter`

```python
from src.proxann.prompter import Prompter

llm_model = "Qwen/Qwen3-8B"  # Must match a model defined in `available_models` for your deployment type (e.g., VLLM, OpenAI)
prompter = Prompter(model_type=llm_model)
```

You can also override configuration parameters such as `temperature`, `max_tokens`, etc., by passing them as keyword arguments. If not specified, defaults are taken from [`config/config.yaml`](config/config.yaml).

#### 2. Make a Prompt Call

```python
result, logprobs = prompter.prompt(system_prompt, question_prompt)
```

- `system_prompt` is optional and can be left as `None`.
- You may also override the `temperature` or other generation parameters at call time.

### **Human Annotations**

User responses were collected through Prolific using the `src.annotations.annotation_server` server. More details on this will be provided soon.

### **Evaluating New LLMs with Human and Topic Model Data**

To obtain LLM annotations, run the `proxann_user_study.py` (or its bash version `bash_scripts/run_proxann.sh`) script with the following parameters:

- **`--model_type "$MODEL_TYPE"`**  
  Specifies the language model(s) to be used for generating annotations. Both open-source and closed-source models are supported. Refer to `config/config.yaml` for the currently available models. New models can be added as needed.

- **`--prompt_mode "$PROMPT_MODE"`**  
  Defines the evaluation steps to perform. Options include:
  - `q1_then_q2_mean`: Step 1 – Category Identification, followed by Step 2 – Relevance Judgment.  
  - `q1_then_q3_mean`: Step 1 – Category Identification, followed by Step 3 – Representativeness Ranking.  
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


    parser.add_argument(
        "--do_both_ways", action="store_true",
        help="Run Q3 twice: once with A as the first document, then reversed.")
    parser.add_argument(
        "--use_user_cats", action="store_true",
        help="Use user categories for Q2/Q3 instead of LLM-generated ones from Q1.",
        default=False)
    parser.add_argument(
        "--removal_condition", type=str,
        default="loose",
        help="Condition for disqualifying responses ('loose': 1+ failures, 'strict': all failures)."
    )
    parser.add_argument(
        "--path_save_results", type=str,
        help="Path to save results.",
        default="data/files_pilot/results")
    parser.add_argument(
        "--temperatures", type=str, default=None,
        help="Temperatures value for the LLM generation in Q1/Q2/Q3, separated by commas."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for random number generator." 
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max tokens for the LLM generation."

### **Reproducing Results from the Paper**


#### Data

- **Datasets:** We use the Wiki and Bills preprocessed datasets from [Hoyle et al. 2022](https://aclanthology.org/2022.findings-emnlp.390/) in their 15,000-term vocabulary form, available in the [original repository](https://github.com/ahoho/topics).

- **Models**: We reuse the 50-topics LDA-Mallet and CTM topic models from [Hoyle et al. 2022](https://aclanthology.org/2022.findings-emnlp.390/) and train a BERTopic model for each dataset using the same experimental setup with default hyperparameters, implemented in the `src.train.tm_trainer.BERTopicTrainer` class. The trained models are available here. 

- **User Study Configuration:** We randomly sample 8 of the 50 topics from the Wiki and Bills datasets for each of the three models. The config files we used for the study are available at XXX and the output that feeds the next two steps (Human Annotations and LLM Annotation) are availabe XXX.

- **User Study Data (Input for Human and LLM annotations):**
  
- **Human annotations:**

- **LLM annotations:**