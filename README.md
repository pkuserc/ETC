<div align="center">

# Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG

<p>
  <strong>English</strong> | <a href="./README_zh.md">简体中文</a>
</p>


<a href="https://arxiv.org/abs/2511.09980"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" /></a>
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/Venue-AAAI%202026-blue" /></a>
[![Task](https://img.shields.io/badge/Task-Dynamic%20RAG-purple.svg)](#overview)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)


**AAAI 2026, Oral**

 <a href="https://deepblue666.github.io/">Bo Li</a>, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang, Wei Ye

</div>

This repository releases the implementation of **ETC** (**E**ntropy-**T**rend **C**onstraint), a training-free method for timely retrieval in dynamic retrieval-augmented generation.

ETC addresses a central problem in dynamic RAG: **when retrieval should be triggered**. Instead of reacting to isolated token-level confidence drops, ETC models the **trend of token-level uncertainty** during generation. It uses the entropy sequence of generated tokens, its first- and second-order differences, and a dynamic smoothing mechanism to detect unstable decoding states earlier and more precisely.

---

## 🌟 Overview

Dynamic RAG improves large language models by retrieving external knowledge only when needed. However, many existing methods trigger retrieval only after the model has already produced low-confidence or incorrect tokens, which leads to **delayed retrieval**.

ETC is designed to solve this problem in a simple and practical way:

1. compute token-level entropy during decoding,
2. build the entropy sequence,
3. measure first-order and second-order entropy differences,
4. smooth the trend signal to reduce noisy triggers,
5. trigger retrieval when uncertainty is rising sharply.

This design allows ETC to inject knowledge at more appropriate positions while reducing unnecessary retrieval operations.

---

## ✨ Highlights

- **Training-free and plug-and-play**  
  ETC does not require additional training or fine-tuning.

- **Trend-aware retrieval timing**  
  ETC models how uncertainty evolves during decoding instead of relying on isolated token confidence.

- **Second-order entropy difference with dynamic smoothing**  
  Retrieval is triggered by sharp changes in uncertainty while suppressing noisy outlier effects.

- **Better timing with fewer retrievals**  
  ETC is designed to improve answer quality while reducing redundant retrieval frequency.

---

## 📦 What Is Released

This repository includes the following files:

- `main.py`  
  Main entry for ETC inference.

- `retriever.py`  
  Retrieval utilities for dynamic RAG.

- `prep_elastic.py`  
  Builds the Elasticsearch index for Wikipedia passages.

- `generate.py`  
  Generation-related utilities.

- `data.py`  
  Dataset loading and preprocessing.

- `evaluate_.py`  
  Evaluation script.

- `config.json`  
  Example runtime configuration.

- `LICENSE`  
  Repository license.

---

## 🗂️ Repository Structure

```text
.
├── LICENSE
├── README.md
├── README_zh.md
├── config.json
├── data.py
├── evaluate_.py
├── generate.py
├── main.py
├── prep_elastic.py
└── retriever.py
```

---

## ⚙️ Installation

We recommend using Python 3.9.

```bash
conda create -n etc python=3.9
conda activate etc
pip install torch==2.1.1 transformers==4.30.2 beir==1.0.1
python -m spacy download en_core_web_sm
```

Depending on your environment, you may also need to install GPU-specific packages separately.

---

## 🧾 Prepare the Retriever

ETC uses a Wikipedia passage collection together with Elasticsearch.

### Step 1. Download Wikipedia passages

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
cd data/dpr
gzip -d psgs_w100.tsv.gz
cd ../..
```

### Step 2. Install and start Elasticsearch

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz
cd elasticsearch-7.17.9
nohup bin/elasticsearch &
cd ../..
```

### Step 3. Build the Wikipedia index

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

---

## 🧪 Datasets

ETC is evaluated on six QA benchmarks:

- 2WikiMultihopQA
- HotpotQA
- StrategyQA
- IIRC
- BioASQ
- PubMedQA

### Download examples

#### 2WikiMultihopQA

Download manually and place the extracted folder under:

```text
data/2wikimultihopqa
```

Reference link:

```text
https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1
```

#### StrategyQA

```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip
```

#### HotpotQA

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

#### IIRC

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```

#### BioASQ

```bash
mkdir -p data/bioasq_7b_yesno
wget -O data/bioasq_7b_yesno/Task7B_yesno_train.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_train.json
wget -O data/bioasq_7b_yesno/Task7B_yesno_validation.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_validation.json
wget -O data/bioasq_7b_yesno/Task7B_yesno_test.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_test.json
```

> Please keep dataset paths consistent with the configuration used by `data.py` and `config.json`.

---

## 🔄 Pipeline

The practical workflow is:

```text
Wikipedia passages
    -> prep_elastic.py
    -> Elasticsearch index

QA dataset
    -> data.py
    -> formatted evaluation samples
    -> main.py
    -> ETC decoding with dynamic retrieval
    -> generate.py / retriever.py
    -> predictions
    -> evaluate_.py
```

---

## 🚀 Quick Start

### Step 1. Configure runtime settings

Edit `config.json` to match your environment, especially fields such as:
- model path
- dataset path
- index name
- retrieval threshold
- backend settings

### Step 2. Run ETC inference

```bash
python main.py --config config.json
```

### Step 3. Evaluate predictions

```bash
python evaluate_.py
```

Before evaluation, ensure the prediction path and reference path are correctly configured.

---

## 🧠 Method Summary

ETC addresses the delayed retrieval problem in dynamic RAG by modeling the **evolution of uncertainty** during decoding.

Instead of triggering retrieval from a single token-level confidence value, ETC tracks:

- the entropy sequence of generated tokens,
- the first-order difference of entropy,
- the second-order difference of entropy,
- a dynamic smoothing signal for robust triggering.

This makes retrieval timing more sensitive to emerging uncertainty trends, enabling earlier and more accurate intervention.

---

## 📊 Main Findings

According to the paper, ETC consistently improves performance over strong training-free RAG baselines across six QA benchmarks and multiple LLM backbones. It is particularly effective in domain-specific scenarios, while also reducing average retrieval count.

These results support the core intuition of ETC: **retrieval timing should be guided by uncertainty trends, not just isolated confidence drops**.

---

## 🛠️ Script Notes

### `main.py`
Main functionality:
- runs ETC-based dynamic RAG inference,
- controls retrieval timing,
- integrates retrieval and generation.

### `retriever.py`
Main functionality:
- performs document retrieval from the external corpus,
- interfaces with the indexed Wikipedia collection.

### `prep_elastic.py`
Main functionality:
- builds the Elasticsearch index for the retriever.

### `generate.py`
Main functionality:
- handles generation-related logic in the decoding loop.

### `data.py`
Main functionality:
- loads and preprocesses evaluation datasets.

### `evaluate_.py`
Main functionality:
- computes evaluation metrics on model predictions.

### `config.json`
Main functionality:
- stores runtime configuration and experiment settings.

---

## ❓ Common Issues

### 1. Elasticsearch is not running
The retriever depends on a working Elasticsearch service and a correctly built Wikipedia index.

### 2. Paths in `config.json` are not updated
Please update all model paths, dataset paths, and index paths before running.

### 3. Dataset files are missing or mislocated
Keep the downloaded datasets under the expected `data/` structure.

### 4. spaCy model is not installed
ETC uses SpaCy for stop-word filtering. Make sure `en_core_web_sm` is installed.

### 5. Evaluation cannot find predictions
Check that the prediction output path matches the path expected by `evaluate_.py`.

---

## 📖 Citation

```bibtex
@inproceedings{li2026modeling,
  title={Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG},
  author={Li, Bo and Tian, Tian and Xu, Zhenghua and Cheng, Hao and Zhang, Shikun and Ye, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={37},
  pages={31527--31535},
  year={2026}
}
```

If you use this repository, please cite the paper.

---
