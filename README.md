<p align="center">
    <h2 align="center">Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG <br>
        [AAAI'26 Oral]
    <br>
     <em>
    <a href="https://deepblue666.github.io/">Bo Li</a>, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang and Wei Ye 
  </em> </h2>
</p>

<p align="justify">
Dynamic retrieval-augmented generation (RAG) enables large language models (LLMs) to fetch external knowledge on demand, improving adaptability over static RAG. A key challenge in this setting is determining when retrieval should occur. Prior methods typically trigger retrieval based on low confidence in individual tokens, which can result in delayed intervention after errors have already occurred. We propose the Entropy-Trend Constraint (ETC), a training-free method that selects optimal retrieval timing by modeling the dynamics of token-level uncertainty. Specifically, ETC leverages first- and second-order differences of the entropy sequence to capture emerging uncertainty trends, enabling earlier and more precise retrieval. Experiments across six QA benchmarks and three LLM backbones show that ETC consistently outperforms strong baselines while reducing retrieval frequency. It is especially effective in domain-specific settings, demonstrating robust generalization. Further ablation studies and qualitative analysis confirm that trend-aware uncertainty modeling leads to more effective retrieval timing. Our approach is plug-and-play, model-agnostic, and easy to integrate into existing decoding pipelines. Code is provided in the supplementary materials.
</p>

- 📖 Paper: [Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG](https://github.com/pkuserc/ETC/tree/main)



## Install environment
```bash
conda create -n etc python=3.9
conda activate etc
pip install torch==2.1.1 transformers==4.30.2 beir==1.0.1
python -m spacy download en_core_web_sm
```
## Prepare Retriever
Followed by dragin. Use the Wikipedia dump and elastic search to build the retriever

### Build Wikipedia index
#### Download the Wikipedia dump
```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

#### Use Elasticsearch to index the Wikipedia dump
```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
```
#### Build the index
```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki 
Download Dataset
```
## Get Dataset
### For 2WikiMultihopQA:
```bash
Download the 2WikiMultihop dataset from its repository https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1. Unzip it and move the folder to data/2wikimultihopqa.
```
### For StrategyQA:
```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip 
```
### For HotpotQA:
```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```
### For IIRC:
```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```
### For BioASQ:
```bash
mkdir -p data/bioasq_7b_yesno
wget -O data/bioasq_7b_yesno/Task7B_yesno_train.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_train.json
wget -O data/bioasq_7b_yesno/Task7B_yesno_validation.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_validation.json
wget -O data/bioasq_7b_yesno/Task7B_yesno_test.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_test.json
```
### For PubMedQA:
```bash
mkdir -p data/pubmedQA
mkdir -p data/pubmedQA
wget -O data/pubmedQA/pqal_train_set.json \
  https://huggingface.co/datasets/tan9/pubmedQA/resolve/main/pqal_train_set.json
wget -O data/pubmedQA/test_set.json \
  https://huggingface.co/datasets/tan9/pubmedQA/resolve/main/test_set.json
```
## Run
The following parameters can be selected in config.json:
| **Parameter** | **Meaning** | **Example / Options** |
|----------------|-------------|------------------------|
| `model_name_or_path` | Hugging Face model | `meta-llama/Llama-2-13b-chat` |
| `dataset` | Dataset | `2wikimultihopqa`, `hotpotqa`, `iirc`, `strategyqa` |
| `data_path` | The folder where the data is located. If you use the above code to download the data, the folder will be `../data/dataset`. | `../data/2wikimultihopqa` |
| `fewshot` | Number of few-shot examples | `6` |
| `sample` | Number of questions sampled from the dataset. `-1` means use the entire dataset. | `1000` |
| `shuffle` | Whether to shuffle the dataset. Without this parameter, the dataset will not be shuffled. | `true`, `false` *(without)* |
| `generate_max_length` | Maximum generated length of a question | `64` |
| `query_formulation` | Way to generate retrieval question. Main: `direct`, `real_words`. Other options: `current_wo_wrong`, `current`, `forward_all`, `last_n_tokens`, `last_sentence` | `real_words` |
| `retrieve_keep_top_k` | Number of reserved tokens when generating a search question | `35` |
| `output_dir` | The generated results will be stored in a folder with a numeric name inside the output folder you specify. If the folder doesn’t exist, it will be created automatically. | `../result/2wikimultihopqa_llama2_13b` |
| `retriever` | Type of retriever | `BM25`, `SGPT` |
| `es_index_name` | Index names in Elasticsearch | `wiki` |
| `retrieve_topk` | Number of related documents retained | `3` |
| `hallucination_threshold` | Threshold at which a word is judged to be incorrect | `1.2` |
| `check_real_words` | Whether only content words participate in threshold judgment. Without this parameter, all words will be considered. | `true`, `false` *(without)* |
| `use_counter` | Whether to count the number of generations, retrievals, questions, tokens, and sentences. Without this parameter, counting will be disabled. | `true`, `false` *(without)* |
| `thres_abs` | Whether to use absolute value for threshold determination | `false` |
Running Example
```bash
python main.py -c path_to_config_file
```
## Evaluation
```bash
python evaluate_.py --dir path_to_folder(result/2wikimultihopqa_llama2_13b/1)
```
