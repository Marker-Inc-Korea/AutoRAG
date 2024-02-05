# AutoRAG

RAG AutoML tool for automatically find optimal RAG pipeline for your date.

[![Release Notes](https://img.shields.io/github/release/Marker-Inc-Korea/AutoRAG)](https://github.com/Marker-Inc-Korea/AutoRAG/releases)
[![Downloads](https://static.pepy.tech/badge/AutoRAG)](https://pepy.tech/project/AutoRAG)

[Document](https://marker-inc-korea.github.io/AutoRAG/)

# Introduction

There are numerous RAG pipelines and modules out there,
but you don’t know what pipeline is great for “your own data” and "your own use-case."
Making and evaluating all RAG modules is very time-consuming and hard to do.
But without it, you will never know which RAG pipeline is the best for your own use-case.

AutoRAG is a tool for finding optimal RAG pipeline for “your data.”
You can evaluate various RAG modules automatically with your own evaluation data,
and find the best RAG pipeline for your own use-case.

AutoRAG supports a simple way to evaluate numerous RAG module combinations.
Try now and find the best RAG pipeline for your own use-case.

# Quick Install

```bash
pip install AutoRAG
```

# Index

- [Introduction](#introduction)
- [Quick Install](#quick-install)
- [Index](#index)
- [Strengths](#strengths)
- [QuickStart](#quickstart)
    - [Prepare your evaluation data](#prepare-your-evaluation-data)
    - [Evaluate your data to various RAG modules](#evaluate-your-data-to-various-rag-modules)
    - [Use a found optimal RAG pipeline](#use-a-found-optimal-rag-pipeline)
    - [Share your RAG pipeline](#share-your-rag-pipeline)
    - [Config yaml file](#config-yaml-file)
- [Supporting RAG modules](#supporting-rag-modules)
- [Roadmap](#roadmap)
- [Contribution](#contribution)

# Strengths

- Find your RAG baseline: Benchmark RAG pipelines with few lines of code. You can quickly get a high-performance RAG
  pipeline just for your data. Don’t waste time dealing with complex RAG modules and academic paper. Focus on your data.
- Analyze where is wrong: Sometimes it is hard to keep tracking where is the major problem within your RAG pipeline.
  AutoRAG gives you the data of it, so you can analyze and focus where is the major problem and where you to focus on.
- Quick Starter Pack for your new RAG product: Get the most effective RAG workflow among many pipelines, and start from
  there. Don’t start at toy-project level, start from advanced level.
- Share your experiment to others: It's really easy to share your experiment to others. Share your config yaml file and
  summary csv files. Plus, check out others result and adapt to your use-case.

# QuickStart

For evaluation, you need to prepare just three files.

- QA dataset file (qa.parquet)
- Corpus dataset file (corpus.parquet)
- Config yaml file (config.yaml)

### Prepare your evaluation data

There is a template for your evaluation data for using AutoRAG.
Check out the evaluation data rule at [here]().
Plus, you can get example datasets for testing AutoRAG at [here](./sample_dataset).

### Evaluate your data to various RAG modules

You can get various config yaml files at [here]().
We highly recommend using pre-made config yaml files for starter.

If you want to make your own config yaml files, check out the [Config yaml file](#config-yaml-file) section.

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

For more details, you can check out how the folder structure looks like at [here]().

### Use a found optimal RAG pipeline

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
Check out API endpoint at [here]().

```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run_api_server()
```

You can run api server with CLI command.

```bash
autorag run_api --config_path your/path/to/pipeline.yaml --host 0.0.0.0 --port 8000
```

### Share your RAG pipeline

You can use your RAG pipeline from extracted pipeline yaml file.
This extracted pipeline is great for sharing your RAG pipeline to others.

You must run this at project folder, which contains datas in data folder, and ingested corpus for retrieval at resources
folder.

```python
from autorag.deploy import extract_best_config

pipeline_dict = extract_best_config(trial_path='your/path/to/trial_folder', output_path='your/path/to/pipeline.yaml')
```

### Config yaml file

You can build your own evaluation process with config yaml file.
You can check detailed explaination how to configure each module and node at [here]().
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
            embedding_model: [ openai, openai_curie ]
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
          metric: [ bleu, meteor ]
        modules:
          - module_type: llama_index_llm
            llm: openai
            model: [ gpt-3.5-turbo-16k, gpt-3.5-turbo-1106 ]
            temperature: [ 0.5, 1.0, 1.5 ]

```

# Supporting RAG modules

### Query Expansion

- Query Decompose
- HyDE

### Retrieval

- BM25
- Vector Retriever (choose your own embedding model)
- Hybrid with rrf (reciprocal rank fusion)
- Hybrid with cc (convex combination)

### Reranker

- UPR
- TART
- MonoT5

### Passage Compressor

- Tree Summarize

### Prompt Maker

- Default Prompt Maker (f-string)

### Generator

- Llama Index LLM (choose your own model)

# Roadmap

- [ ] Policy Module for modular RAG pipeline
- [ ] Visualize evaluation result
- [ ] Visualize config yaml file
- [ ] More RAG modules support
- [ ] Token usage strategy
- [ ] Multi-modal support
- [ ] More evaluation metrics
- [ ] Answer Filtering Module
- [ ] Optimization checkpoint (Auto-Save)

# Contribution

We are developing AutoRAG as open-source. Feel free to contribute to this project.
If you want to contribute to this project, please check out the [Contribution Guide](CONTRIBUTING.md).
Plus, check out our detailed documentation at [here]().
