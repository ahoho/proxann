<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="./figures/repo/Proxann1.png" alt="Logo" width="60" height="60" style="vertical-align: middle; margin-right: 10px;">
  <span style="font-size: 2em;">THETA-EVALUATION</span>
</p>
<!-- markdownlint-enable MD033 -->
<!-- markdownlint-disable MD041 -->

This repository contains the code and data for reproducing experiments from our paper, *ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering.*

## Features

1. **Human User Study Generation**:
   - Use the script `get_user_study_data.py` to generate a JSON file for conducting user studies.
2. **Proxy-Based Evaluation**:
   - Run LLM proxy annotations using `run_proxy_ann_study.py`.
3. **Topic Model Training**:
   - Train topic models for evaluation using the `TMTrainer` class.

## Data

TODO: Update (make links and remove full path)

User responses: 

- /data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv
- /data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv

Model information:

- /data/json_out/config_pilot_wiki.json and /data/json_out/config_pilot_wiki_part2.json
- /data/json_out/config_bills_part1.json

generated with the user study configuration files:

- config/user_study/config_pilot_bills.conf
- config/user_study/config_pilot_wiki.conf

and the topic models, which are available upon request.

## Installation

## Use

> **IMPORTANT TO RUN LDA-Mallet**
>
> Download the [latest release of Mallet](https://github.com/mimno/Mallet/releases) and place it in the `src/train` directory. This can be done using the script `bash_scripts/wget_mallet.sh`.