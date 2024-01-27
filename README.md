# AutoRAG
## ⚠️Warning: This repository is under construction. 🚧 

# Introduction

There are numerous RAG methods and modules out there,
but you don’t know what method is great for “your own data” and "your own use-case."
Making and evaluating all RAG modules is very time-consuming and hard to do.
But without it, you will never know your own RAG pipeline is “fit” for your own use-case than other RAG methods.

AutoRAG is a tool for benchmarking “your data” to conventional RAG methods. 
You can build evaluation dataset from your raw documents, and evaluate various RAG methods automatically.
With that result, you can quickly find what is the best RAG pipeline to your own data.

# Index
- [Introduction](#introduction)
- [Index](#index)
- [Strengths](#strengths)
- [QuickStart](#quickstart)
  - [Generate synthetic evaluation dataset](#generate-synthetic-evaluation-dataset)
  - [Evaluate your data to various RAG modules](#evaluate-your-data-to-various-rag-modules)
  - [Evaluate your custom RAG pipeline](#evaluate-your-custom-rag-pipeline)
  - [Config yaml file](#config-yaml-file)
- [Installation](#installation)
- [Supporting RAG modules](#supporting-rag-modules)
- [To-do List](#to-do-list)
- [Contribution](#contribution)

# Strengths
- Data Creation: Make your raw documents to RAG evaluation dataset. Generate multi-hop, ambiguous, conversational, non-answerable questions, just like real-world user questions.
- Find your RAG baseline: Easily benchmark 30+ RAG methods with few lines of code. You can quickly get a high-performance RAG pipeline just for your data. Don’t waste time dealing with complex RAG modules and academic paper. Focus on your data.
- Analyze where is wrong: Sometimes it is hard to keep tracking where is the major problem within your RAG pipeline. AutoRAG gives you the data of it, so you can analyze and focus where is the major problem and where you to focus on.
- Quick Starter Pack for your new RAG product: Get the most effective RAG workflow among many pipelines, and start from there. Don’t start at toy-project level, start from advanced level.
- Share your experiment to others: It's really easy to share your experiment to others. Share your config yaml file and evaluation result parquet files. Plus, check out others result and adapt to your use-case.

# QuickStart

### Generate synthetic evaluation dataset
```python
data = pd.read_parquet('your/data.parquet')
generator = DataGenerator(prompt="This data is news dataset, cralwed from finance news site. You need to make detailed question about finance news. Do not make questions that not relevant to economy or finance domain.")
evaluate_dataset = generator.generate(data)
evaluate_dataset.to_parquet('your/path/to/evaluate_dataset.parquet')
```

### Evaluate your data to various RAG modules
```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(qa_data_path='your/path/to/qa.parquet', corpus_data_path='your/path/to/corpus.parquet')
evaluator.start_trial('your/path/to/config.yaml')
```
or you can use command line interface
```bash
autorag evaluate --config your/path/to/default_config.yaml --qa_data_path your/path/to/qa.parquet --corpus_data_path your/path/to/corpus.parquet
```

### Use a found optimal RAG pipeline
You can use a found optimal RAG pipeline right away.
It needs just a few lines of code, and you are ready to use!

First, you need to build pipeline yaml file from your evaluated trial folder.

```python
from autorag.deploy import extract_best_config

pipeline_dict = extract_best_config(trial_path='your/path/to/trial_folder', output_path='your/path/to/pipeline.yaml')
```

Then, you can use your RAG pipeline from extracted pipeline yaml file.
Plus, you can share your RAG pipeline to others just by sharing pipeline yaml file.
You must run this at project folder.
It will automatically find ingested corpus for retrieval and fetching data for RAG system.
```python
from autorag.deploy import Runner

runner = Runner.from_yaml('your/path/to/pipeline.yaml')
runner.run('your question')
```

Or run from a trial folder that you want to run.
```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run('your question')
```

Or, you can run this pipeline as api server.
You can use python code or CLI command.
```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run_api_server()
```

```bash
autorag run_api --config_path your/path/to/pipeline.yaml --host 0.0.0.0 --port 8000
```


### Evaluate your custom RAG pipeline

```python
from autorag.evaluate import evaluate_retrieval, evaluate_generation

@evaluate
def your_custom_rag_pipeline(query: str) -> str:
    # your custom rag pipeline
    return answer


# also, you can evaluate each RAG module one by one
@evaluate_retrieval(retrieval_gt=retrieval_gt, metrics=['retrieval_f1', 'retrieval_recall', 'retrieval_precision'])
def your_retrieval_module(query: str, top_k: int = 5) ->  tuple[list[list[str]], list[list[str]], list[list[float]]]:
    # your custom retrieval module
    return retrieved_contents, scores, retrieved_ids

@evaluate_generation(generation_gt=generation_gt, metrics=['bleu', 'rouge'])
def your_llm_module(prompt: str) -> list[str]:
    # your custom llm module
    return answers
```

### Config yaml file
You can build your own evaluation process with config yaml file.
There is a simple yaml file example.
It evaluates two retrieval modules which are BM25 and Vector Retriever, and three reranking modules.
Lastly, it generates prompt and makes generation with three other LLM models. 
```yaml
qa_dataset_path: your/path/to/qa.parquet
corpus_path: your/path/to/corpus.parquet
node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: retrieval
      strategy:
        metric: f1
      top_k: 50
      modules:
        - module_type: bm25
        - module_type: vector
          embedding_model: [openai, huggingface]
    - node_type: reranker
      strategy:
        metric: precision
        speed_threshold: 5
      top_k: 3
      modules:
        - module_type: upr
        - module_type: tart
          prompt: Arrange the following sentences in the correct order.
        - module_type: monoT5
- node_line_name: 
  nodes:
    - node_type: prompt_maker
      skip_evaluation: True
      modules:
        - module_type: default_prompt_maker
          prompt: "This is a news dataset, cralwed from finance news site. You need to make detailed question about finance news. Do not make questions that not relevant to economy or finance domain.\n{{context}}\n\nQ: {{question}}\nA:"
    - node_type: generation
      strategy:
        metric: [bleu, rouge, kf1]
      modules:
        - module_type: openai
        - module_type: mixtral
        - module_type: llama-2
```

# Installation
Warning! You can't install autorag yet because it is under construction.
```bash
pip install autorag
```

## Installation for developing
```bash
pip install -e .
pip install -r dev_requirements.txt
```

Then, You can test with pytest.

```bash
pytest
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
- Summarizer

### Prompt Maker
- Default Prompt Maker (f-string)
- LLMLingua

### Generation
Choose your own model

# To-do List
- [ ] Custom yaml config file for each use-case
- [ ] Existing QA dataset evaluation
- [ ] Policy Module for modular RAG pipeline
- [ ] Visualize evaluation result
- [ ] Visualize config yaml file
- [ ] More RAG modules
- [ ] Human Dataset Creation Helper

# Contribution
We are developing AutoRAG as open-source. Feel free to contribute to this project.
