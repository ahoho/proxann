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

### Configuration

#### 1. LLMs

##### GPT Models (OpenAI)

To use GPT models via the OpenAI API, create a `.env` file in the root directory with the following content:
```bash
OPENAI_API_KEY=[your_open_ai_api_key]
```
You can also modify the path to the ``.env`` file in the [configuration file](config/config.yaml).

##### Open-Source Models (via vLLM)
We rely on [vLLM models](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai) for evaluating with open-source large language models. You must have the model running and specify the endpoint where it is deployed in the [configuration file](config/config.yaml).

#### 2. Topics Models

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

<div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 12px; border-radius: 6px; margin-bottom: 1em;"> <strong>Recommendation:</strong> For best overall alignment with human judgments, <strong>GPT-4o</strong> and <strong>Qwen 2.5–72B</strong> perform the strongest across both Fit and Rank steps. <strong>Qwen 1.5–32B</strong> is a solid cost-effective alternative. Avoid <strong>Llama 3.1–8B</strong>, which consistently underperforms. </div>

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


### **Preparing Topic Models for ProxAnn**


### **Evaluating Topic Model Estimates with ProxAnn Metrics**


### **Running LLMs Independently (Prompter)**



### **End-to-End Example: A Practical Walkthrough**


### **Evaluating New LLMs with Human and Topic Model Data**


### **Reproducing Results from the Paper**

