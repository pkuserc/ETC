<div align="center">

# Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG

<p>
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

<a href="https://arxiv.org/abs/2511.09980"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" /></a>
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/Venue-AAAI%202026-blue" /></a>
[![Task](https://img.shields.io/badge/Task-Dynamic%20RAG-purple.svg)](#overview)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)


**AAAI 2026, Oral**

 <a href="https://deepblue666.github.io/">Bo Li</a>, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang, Wei Ye

</div>

本仓库公开了 **ETC**（**E**ntropy-**T**rend **C**onstraint）的实现代码。ETC 是一个面向动态检索增强生成的 **training-free** 方法，用于解决动态 RAG 中一个关键问题：**检索究竟应该在何时触发**。

与依赖单个 token 置信度进行触发的做法不同，ETC 建模的是生成过程中 **token-level uncertainty 的变化趋势**。它基于 token 熵序列及其一阶、二阶差分，并结合动态平滑机制，更早、更稳定地检测模型即将进入不稳定生成区域的时刻，从而在更合适的时间注入外部知识。

> Related RAG projects from us: [**GRIP** (ACL 2026 Main)](https://github.com/WisdomShell/GRIP) · [**ETC** (AAAI 2026 Oral)](https://github.com/WisdomShell/ETC) · [**SCD** (AAAI 2026 Oral)](https://github.com/WisdomShell/SCD)
---

## 🌟 概述

动态 RAG 允许模型在生成过程中按需检索外部知识，但很多已有方法只有在模型已经生成了低置信度甚至错误 token 后才触发检索，这就导致了 **delayed retrieval**。

ETC 旨在用一个简单而实用的策略解决这个问题：

1. 在解码过程中计算 token-level entropy，
2. 构造 entropy sequence，
3. 计算熵序列的一阶和二阶差分，
4. 用动态平滑抑制噪声触发，
5. 在不确定性快速上升时触发检索。

这种设计既能让外部知识在更合理的位置被引入，也能减少不必要的检索次数。

---

## ✨ 亮点

- **Training-free and plug-and-play**  
  ETC 不需要额外训练或微调。

- **Trend-aware retrieval timing**  
  ETC 关注不确定性如何随生成过程演化，而不是只依赖某个单独 token 的置信度。

- **Second-order entropy difference with dynamic smoothing**  
  方法利用二阶熵差分检测不稳定趋势，并通过动态平滑降低异常值带来的误触发。

- **Better timing with fewer retrievals**  
  ETC 旨在提升答案质量，同时减少冗余检索。

---

## 📦 公开内容

本仓库包含以下文件：

- `main.py`  
  ETC 推理主入口。

- `retriever.py`  
  动态 RAG 的检索模块工具。

- `prep_elastic.py`  
  为 Wikipedia passages 建立 Elasticsearch 索引。

- `generate.py`  
  与生成过程相关的辅助逻辑。

- `data.py`  
  数据集加载与预处理。

- `evaluate_.py`  
  评测脚本。

- `config.json`  
  运行配置示例。

- `LICENSE`  
  仓库许可证。

---

## 🗂️ 仓库结构

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

## ⚙️ 安装

建议使用 Python 3.9。

```bash
conda create -n etc python=3.9
conda activate etc
pip install torch==2.1.1 transformers==4.30.2 beir==1.0.1
python -m spacy download en_core_web_sm
```

根据你的环境，可能还需要额外安装 GPU 相关依赖。

---

## 🧾 检索器准备

ETC 使用 Wikipedia passage 语料与 Elasticsearch 构建检索器。

### 第一步：下载 Wikipedia passages

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
cd data/dpr
gzip -d psgs_w100.tsv.gz
cd ../..
```

### 第二步：安装并启动 Elasticsearch

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz
cd elasticsearch-7.17.9
nohup bin/elasticsearch &
cd ../..
```

### 第三步：建立 Wikipedia 索引

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

---

## 🧪 数据集

ETC 在以下六个 QA benchmark 上进行了评测：

- 2WikiMultihopQA
- HotpotQA
- StrategyQA
- IIRC
- BioASQ
- PubMedQA

### 下载示例

#### 2WikiMultihopQA

请手动下载，并将解压后的目录放到：

```text
data/2wikimultihopqa
```

参考链接：

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

> 请保持下载后的数据路径与 `data.py` 和 `config.json` 中的设置一致。

---

## 🔄 方法流程

实际运行流程如下：

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

## 🚀 快速开始

### 第一步：配置运行参数

首先修改 `config.json`，重点包括：
- 模型路径
- 数据集路径
- 索引名称
- 检索阈值
- 后端相关设置

### 第二步：运行 ETC 推理

```bash
python main.py --config config.json
```

### 第三步：评测预测结果

```bash
python evaluate_.py
```

运行评测前，请确保预测文件路径和参考答案路径已经正确配置。

---

## 🧠 方法总结

ETC 通过建模生成过程中 **不确定性的演化趋势** 来解决动态 RAG 中的 delayed retrieval 问题。

与其根据单个 token 的置信度决定是否检索，ETC 追踪的是：

- token entropy sequence，
- entropy 的一阶差分，
- entropy 的二阶差分，
- 用于稳定触发的动态平滑信号。

因此，它能更敏感地捕捉模型即将进入低置信度区域的时刻，从而更早、更准确地触发检索。

---

## 📊 主要结论

根据论文实验，ETC 在六个 QA benchmark 和多个 LLM backbone 上都稳定优于强基线 training-free RAG 方法，并且平均检索次数更少。

这些结果支持了 ETC 的核心观点：**检索时机应由不确定性趋势来指导，而不只是依赖某个孤立 token 的置信度下降。**

---

## 🛠️ 脚本说明

### `main.py`
主要功能：
- 执行基于 ETC 的动态 RAG 推理，
- 控制检索触发时机，
- 集成检索与生成过程。

### `retriever.py`
主要功能：
- 从外部语料中执行文档检索，
- 与 Wikipedia 索引交互。

### `prep_elastic.py`
主要功能：
- 建立 Elasticsearch 检索索引。

### `generate.py`
主要功能：
- 提供解码过程中的生成辅助逻辑。

### `data.py`
主要功能：
- 加载并预处理评测数据集。

### `evaluate_.py`
主要功能：
- 对模型预测结果进行评测。

### `config.json`
主要功能：
- 保存运行配置与实验设置。

---

## ❓ 常见问题

### 1. Elasticsearch 没有正常启动
检索模块依赖可用的 Elasticsearch 服务以及正确建立的 Wikipedia 索引。

### 2. `config.json` 中路径未更新
请在运行前检查所有模型路径、数据路径和索引路径。

### 3. 数据集缺失或路径错误
请确保所有下载的数据集都放在预期的 `data/` 目录结构下。

### 4. spaCy 模型未安装
ETC 依赖 SpaCy 做停用词过滤，请确认 `en_core_web_sm` 已安装。

### 5. 评测脚本找不到预测结果
请检查预测输出路径是否与 `evaluate_.py` 的预期一致。

---

## 🧭 Related RAG Projects

本仓库属于我们围绕 **可控、可适应 Retrieval-Augmented Generation（RAG）** 展开的系列研究工作之一。

- **GRIP** [ACL 2026 Main Conference]: [Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning](https://github.com/WisdomShell/GRIP)  
  一个 **training-based 的动态 RAG** 框架，将检索控制内化到 token-level decoding 中。  

- **ETC** [AAAI 2026 Oral Paper]: [Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG](https://github.com/WisdomShell/ETC)  
  一个 **training-free 的动态 RAG** 方法，重点通过建模解码过程中的熵趋势来改进检索时机。  

- **SCD** [AAAI 2026 Oral Paper]: [Language Drift in Multilingual Retrieval-Augmented Generation](https://github.com/WisdomShell/SCD)  
  一个 **training-free 的多语言 RAG** 方法，通过 decoding-time control 缓解语言漂移问题。  

这些项目共同覆盖了 RAG 的三个互补方向：  
**training-based retrieval planning、training-free retrieval timing，以及 multilingual generation 的 decoding-time control**。

## 📖 引用

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

---
