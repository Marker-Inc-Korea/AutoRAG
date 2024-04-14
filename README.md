# AutoRAG

RAG AutoML tool for automatically finds an optimal RAG pipeline for your data.

Explore our 📖 [Document](https://marker-inc-korea.github.io/AutoRAG/)!!

Plus, join our 📞 [Discord](https://discord.gg/P4DYXfmSAs) Community.

---

📌 Colab Tutorial

- [Step 1: Basic of AutoRAG | Optimizing your RAG pipeline](https://colab.research.google.com/drive/19OEQXO_pHN6gnn2WdfPd4hjnS-4GurVd?usp=sharing)
- [Step 2: Create evaluation dataset](https://colab.research.google.com/drive/1HXjVHCLTaX7mkmZp3IKlEPt0B3jVeHvP#scrollTo=cgFUCuaUZvTr)

---
🚨 YouTube Tutorial

https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/c0d23896-40c0-479f-a17b-aa2ec3183a26

_Muted by default, enable sound for voice-over_

You can see on [YouTube](https://youtu.be/2ojK8xjyXAU?feature=shared)

# 📑 Index

- [Introduction](#introduction)
- [Quick Install](#-quick-install)
- [Index](#-index)
- [Strengths](#-strengths)
- [QuickStart](#-quickstart)
  - [1. Prepare your evaluation data](#1-prepare-your-evaluation-data)
  - [2. Evaluate your data to various RAG modules](#2-evaluate-your-data-to-various-rag-modules)
  - [3. Use a found optimal RAG pipeline](#3-use-a-found-optimal-rag-pipeline)
  - [4. Share your RAG pipeline](#4-share-your-rag-pipeline)
  - [+ Config yaml file](#-create-your-own-config-yaml-file)
- [Supporting RAG modules](#supporting-nodes--modules)
- [Roadmap](#roadmap)
- [Contribution](#contribution)

# Introduction

There are many RAG pipelines and modules out there,
but you don’t know what pipeline is great for “your own data” and "your own use-case."
Making and evaluating all RAG modules is very time-consuming and hard to do.
But without it, you will never know which RAG pipeline is the best for your own use-case.

AutoRAG is a tool for finding optimal RAG pipeline for “your data.”
You can evaluate various RAG modules automatically with your own evaluation data,
and find the best RAG pipeline for your own use-case.

AutoRAG supports a simple way to evaluate many RAG module combinations.
Try now and find the best RAG pipeline for your own use-case.

# ⚡ Quick Install

```bash
pip install AutoRAG
```

# 💪 Strengths

### **1. Find your RAG baseline**

Benchmark RAG pipelines with few lines of code. You can quickly get a high-performance RAG
pipeline just for your data. Don’t waste time dealing with complex RAG modules and academic paper. Focus on your data.

### **2. Analyze where is wrong**

Sometimes it is hard to keep tracking where is the major problem within your RAG pipeline.
AutoRAG gives you the data of it, so you can analyze and focus where is the major problem and where you to focus on.

### **3. Quick Starter Pack for your new RAG product**

Get the most effective RAG workflow among many pipelines, and start from
there.
Don’t start at toy-project level, start from advanced level.

### **4. Share your experiment to others**

It's really easy to share your experiment to others. Share your config yaml file and
summary csv files. Plus, check out others result and adapt to your use-case.

# ⚡ QuickStart

### 1. Prepare your evaluation data

For evaluation, you need to prepare just three files.

- QA dataset file (qa.parquet)
- Corpus dataset file (corpus.parquet)
- Config yaml file (config.yaml)

There is a template for your evaluation data for using AutoRAG.

- Check out how to make evaluation data
  at [here](https://marker-inc-korea.github.io/AutoRAG/data_creation/tutorial.html).
- Check out the evaluation data rule
  at [here](https://marker-inc-korea.github.io/AutoRAG/data_creation/data_format.html).
- Plus, you can get example datasets for testing AutoRAG at [here](./sample_dataset).

### 2. Evaluate your data to various RAG modules

You can get various config yaml files at [here](./sample_config).
We highly recommend using pre-made config yaml files for starter.

If you want to make your own config yaml files, check out the [Config yaml file](#-create-your-own-config-yaml-file)
section.

You can evaluate your RAG pipeline with just a few lines of code.

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(qa_data_path='your/path/to/qa.parquet', corpus_data_path='your/path/to/corpus.parquet')
evaluator.start_trial('your/path/to/config.yaml')
```

or you can use command line interface

```bash
autorag evaluate --config your/path/to/default_config.yaml --qa_data_path your/path/to/qa.parquet --corpus_data_path your/path/to/corpus.parquet
```

Once it is done, you can see several files and folders created at your current directory.
At the trial folder named to numbers (like 0),
you can check `summary.csv` file that summarizes the evaluation results and the best RAG pipeline for your data.

For more details, you can check out how the folder structure looks like
at [here](https://marker-inc-korea.github.io/AutoRAG/optimization/folder_structure.html).

### 3. Use a found optimal RAG pipeline

You can use a found optimal RAG pipeline right away.
It needs just a few lines of code, and you are ready to use!

First, you need to build pipeline yaml file from your evaluated trial folder.
You can find the trial folder in your current directory.
Just looking folder like '0' or other numbers.

```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run('your question')
```

Or, you can run this pipeline as api server.
You can use python code or CLI command.
Check out API endpoint at [here](https://marker-inc-korea.github.io/AutoRAG/deploy/api_endpoint.html).

```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run_api_server()
```

You can run api server with CLI command.

```bash
autorag run_api --config_path your/path/to/pipeline.yaml --host 0.0.0.0 --port 8000
```

### 4. Share your RAG pipeline

You can use your RAG pipeline from extracted pipeline yaml file.
This extracted pipeline is great for sharing your RAG pipeline to others.

You must run this at project folder, which contains datas in data folder, and ingested corpus for retrieval at resources
folder.

```python
from autorag.deploy import extract_best_config

pipeline_dict = extract_best_config(trial_path='your/path/to/trial_folder', output_path='your/path/to/pipeline.yaml')
```

### ➕ Create your own Config yaml file

You can build your own evaluation process with config yaml file.
You can check detailed explanation how to configure each module and node
at [here](https://marker-inc-korea.github.io/AutoRAG/nodes/index.html#).

There is a simple yaml file example.

It evaluates two retrieval modules which are BM25 and Vector Retriever, and three reranking modules.
Lastly, it generates prompt and makes generation with two other LLM models and three temperatures.

```yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: retrieval
        strategy:
          metric: retrieval_f1
        top_k: 50
        modules:
          - module_type: bm25
          - module_type: vector
            embedding_model: [ openai, openai_embed_3_large ]
          - module_type: hybrid_rrf
            target_modules: ('bm25', 'vectordb')
            rrf_k: [ 3, 5, 10 ]
      - node_type: reranker
        strategy:
          metric: retrieval_precision
          speed_threshold: 5
        top_k: 3
        modules:
          - module_type: upr
          - module_type: tart
            prompt: Arrange the following sentences in the correct order.
          - module_type: monoT5
  - node_line_name: generate_node_line
    nodes:
      - node_type: prompt_maker
        modules:
          - module_type: fstring
            prompt: "This is a news dataset, crawled from finance news site. You need to make detailed question about finance news. Do not make questions that not relevant to economy or finance domain.\n{retrieved_contents}\n\nQ: {query}\nA:"
      - node_type: generator
        strategy:
          metric:
            - metric_name: meteor
            - metric_name: rouge
            - metric_name: sem_score
              embedding_model: openai
            - metric_name: g_eval
              model: gpt-3.5-turbo
        modules:
          - module_type: llama_index_llm
            llm: openai
            model: [ gpt-3.5-turbo-16k, gpt-3.5-turbo-1106 ]
            temperature: [ 0.5, 1.0, 1.5 ]

```

# ❗Supporting Nodes & modules

You can check our all supporting Nodes & modules
at [here](https://edai.notion.site/Supporting-Nodes-modules-0ebc7810649f4e41aead472a92976be4?pvs=4)

# 🛣Roadmap

- [ ] Policy Module for modular RAG pipeline
- [ ] Visualize evaluation result
- [ ] Visualize config yaml file
- [ ] More RAG modules support
- [x] Token usage strategy
- [ ] Multi-modal support
- [ ] More evaluation metrics
- [ ] Answer Filtering Module
- [x] Restart optimization from previous trial

# Contribution

We are developing AutoRAG as open-source.

So this project welcomes contributions and suggestions. Feel free to contribute to this project.

Plus, check out our detailed documentation at [here](https://marker-inc-korea.github.io/AutoRAG/index.html).
