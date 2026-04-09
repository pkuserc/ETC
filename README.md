<div align="center">
 
**ETC: Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16463)
[![Venue](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![Task](https://img.shields.io/badge/Task-Dynamic%20RAG-purple.svg)](#overview)
 
**AAAI 2026, Oral**

 <a href="https://deepblue666.github.io/">Bo Li</a>, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang, Wei Ye

</div>

---

## Overview

Dynamic retrieval-augmented generation (Dynamic RAG) allows large language models to retrieve external knowledge on demand during generation. A central challenge is **when retrieval should happen**.

Many existing methods trigger retrieval based on the confidence of individual tokens. This can lead to **delayed retrieval**, where the model has already drifted before external evidence is introduced.

**ETC** is a **training-free** method for timely retrieval in Dynamic RAG. Instead of relying on isolated token confidence, ETC models the **trend of token-level uncertainty** during decoding. It uses:

- first-order differences of the entropy sequence
- second-order differences of the entropy sequence
- a dynamic smoothing strategy for more stable triggering

This allows retrieval to be activated **earlier and more precisely**, while also reducing unnecessary retrievals.

### Key Features

- **Training-free** and plug-and-play
- **Model-agnostic**
- Easy to integrate into existing decoding pipelines
- Evaluated on multiple QA benchmarks and LLM backbones
- Improves answer quality while reducing retrieval frequency

---

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── config.json
├── data.py
├── evaluate_.py
├── generate.py
├── main.py
├── prep_elastic.py
└── retriever.py
```

### File Description

- `main.py`: main entry for ETC inference
- `retriever.py`: retrieval utilities
- `prep_elastic.py`: build the Elasticsearch index for Wikipedia passages
- `generate.py`: generation-related utilities
- `data.py`: dataset loading and preprocessing
- `evaluate_.py`: evaluation script
- `config.json`: example runtime configuration

---

## Method Summary

ETC addresses the delayed retrieval problem in Dynamic RAG by modeling the **evolution of uncertainty during decoding**.

Instead of making retrieval decisions from a single token score, ETC tracks the entropy sequence across generated tokens and identifies unstable generation trends. Retrieval is triggered when the model is likely entering a low-confidence region, before errors accumulate.

ETC also introduces a **dynamic smoothing mechanism** to reduce noisy triggers and redundant retrieval caused by local entropy fluctuations.

---

## Installation

We recommend using **Python 3.9**.

```bash
conda create -n etc python=3.9
conda activate etc
pip install torch==2.1.1 transformers==4.30.2 beir==1.0.1
python -m spacy download en_core_web_sm
```

---

## Prepare the Retriever

This repository uses a Wikipedia passage collection together with Elasticsearch to build the retriever.

### 1. Download Wikipedia Passages

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

### 2. Install and Start Elasticsearch

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz
cd elasticsearch-7.17.9
nohup bin/elasticsearch &
```

### 3. Build the Wikipedia Index

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

---

## Datasets

ETC is evaluated on the following QA benchmarks:

- **2WikiMultihopQA**
- **HotpotQA**
- **StrategyQA**
- **IIRC**
- **BioASQ**
- **PubMedQA**

### Download Instructions

#### 2WikiMultihopQA

Download the dataset manually from its official repository, then unzip it and move the folder to:

```text
data/2wikimultihopqa
```

Reference download link:

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

#### PubMedQA

```bash
mkdir -p data/pubmedQA
wget -O data/pubmedQA/pqal_train_set.json \
  https://huggingface.co/datasets/tan9/pubmedQA/resolve/main/pqal_train_set.json
wget -O data/pubmedQA/test_set.json \
  https://huggingface.co/datasets/tan9/pubmedQA/resolve/main/test_set.json
```

---

## Configuration

Main runtime options are specified in `config.json`.

### Important Arguments

| Argument | Description | Example |
|---|---|---|
| `model_name_or_path` | Hugging Face model path | `meta-llama/Llama-2-13b-chat` |
| `dataset` | Dataset name | `2wikimultihopqa`, `hotpotqa`, `iirc`, `strategyqa` |
| `data_path` | Dataset directory | `../data/2wikimultihopqa` |
| `fewshot` | Number of few-shot examples | `6` |
| `sample` | Number of sampled questions. `-1` means all data | `1000` |
| `shuffle` | Whether to shuffle the dataset | `true`, `false` |
| `generate_max_length` | Maximum generated query length | `64` |
| `query_formulation` | Retrieval query generation strategy | `direct`, `real_words`, `current`, `last_sentence` |
| `retrieve_keep_top_k` | Number of reserved tokens for query construction | `35` |
| `output_dir` | Output directory for results | `../result/2wikimultihopqa_llama2_13b` |
| `retriever` | Retriever type | `BM25`, `SGPT` |
| `es_index_name` | Elasticsearch index name | `wiki` |

---

## Quick Start

After preparing the retriever and datasets:

```bash
python main.py -c config.json
```

You can also run with another config file:

```bash
python main.py -c path_to_config_file
```

---

## Evaluation

Run the evaluation script with:

```bash
python evaluate_.py
```

---

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{DBLP:conf/aaai/LiTXCZY26,
  author       = {Bo Li and
                  Tian Tian and
                  Zhenghua Xu and
                  Hao Cheng and
                  Shikun Zhang and
                  Wei Ye},
  title        = {Modeling Uncertainty Trends for Timely Retrieval in Dynamic {RAG}},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {31527--31535},
  publisher    = {{AAAI} Press},
  year         = {2026},
}
```
