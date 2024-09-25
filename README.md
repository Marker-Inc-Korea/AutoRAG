# AutoRAG

RAG AutoML tool for automatically finds an optimal RAG pipeline for your data.

![Thumbnail](https://github.com/user-attachments/assets/7b3bdd01-709c-4579-b9b6-152c34a390de)

There are many RAG pipelines and modules out there,
but you don’t know what pipeline is great for “your own data” and "your own use-case."
Making and evaluating all RAG modules is very time-consuming and hard to do.
But without it, you will never know which RAG pipeline is the best for your own use-case.

AutoRAG is a tool for finding optimal RAG pipeline for “your data.”
You can evaluate various RAG modules automatically with your own evaluation data,
and find the best RAG pipeline for your own use-case.

AutoRAG supports a simple way to evaluate many RAG module combinations.
Try now and find the best RAG pipeline for your own use-case.

Explore our 📖 [Document](https://docs.auto-rag.com)!!

Plus, join our 📞 [Discord](https://discord.gg/P4DYXfmSAs) Community.

---

## YouTube Tutorial

https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/c0d23896-40c0-479f-a17b-aa2ec3183a26

_Muted by default, enable sound for voice-over_

You can see on [YouTube](https://youtu.be/2ojK8xjyXAU?feature=shared)

## Colab Tutorial

- [Step 1: Basic of AutoRAG | Optimizing your RAG pipeline](https://colab.research.google.com/drive/19OEQXO_pHN6gnn2WdfPd4hjnS-4GurVd?usp=sharing)


# Quick Install

We recommend using Python version 3.10 or higher for AutoRAG.

```bash
pip install AutoRAG
```

# Data Creation

![image](https://github.com/user-attachments/assets/5c86a66d-9c99-4a51-a307-8b12796d029f)

RAG Optimization requires two types of data: QA dataset and Corpus dataset.

1. **QA** dataset file (qa.parquet)
2. **Corpus** dataset file (corpus.parquet)

**QA** dataset is important for accurate and reliable evaluation and optimization.

**Corpus** dataset is critical to the performance of RAGs.
This is because RAG uses the corpus to retrieve documents and generate answers using it.

## Quick Start

### 1. Parsing

#### Set YAML File

```yaml
modules:
  - module_type: langchain_parse
    parse_method: pdfminer
```

You can also use multiple Parse modules at once.
However, in this case, you'll need to return a new process for each parsed result.

#### Start Parsing

You can parse your raw documents with just a few lines of code.

```python
from autorag.parser import Parser

parser = Parser(data_path_glob="your/data/path/*")
parser.start_parsing("your/path/to/parse_config.yaml")
```

### 2. Chunking

#### Set YAML File

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: Token
    chunk_size: 1024
    chunk_overlap: 24
    add_file_name: english
```

You can also use multiple Chunk modules at once.
In this case, you need to use one corpus to create QA, and then map the rest of the corpus to QA Data.
If the chunk method is different, the retrieval_gt will be different, so we need to remap it to the QA dataset.

#### Start Chunking

You can chunk your parsed results with just a few lines of code.

```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet(parsed_data_path="your/parsed/data/path")
chunker.start_chunking("your/path/to/chunk_config.yaml")
```

### 3. QA Creation

You can create QA dataset with just a few lines of code.

```python
import pandas as pd
from llama_index.llms.openai import OpenAI

from autorag.data.beta.filter.dontknow import dontknow_filter_rule_based
from autorag.data.beta.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt,
)
from autorag.data.beta.schema import Raw, Corpus
from autorag.data.beta.query.llama_gen_query import factoid_query_gen
from autorag.data.beta.sample import random_single_hop

llm = OpenAI()
raw_df = pd.read_parquet("your/path/to/corpus.parquet")
raw_instance = Raw(raw_df)

corpus_df = pd.read_parquet("your/path/to/corpus.parquet")
corpus_instance = Corpus(corpus_df, raw_instance)

initial_qa = (
    corpus_instance.sample(random_single_hop, n=3)
    .map(
        lambda df: df.reset_index(drop=True),
    )
    .make_retrieval_gt_contents()
    .batch_apply(
        factoid_query_gen,  # query generation
        llm=llm,
    )
    .batch_apply(
        make_basic_gen_gt,  # answer generation (basic)
        llm=llm,
    )
    .batch_apply(
        make_concise_gen_gt,  # answer generation (concise)
        llm=llm,
    )
    .filter(
        dontknow_filter_rule_based,  # filter don't know
        lang="en",
    )
)

initial_qa.to_parquet('./qa.parquet', './corpus.parquet')
```

# RAG Optimization
![rag](https://github.com/user-attachments/assets/b4f48144-2866-46e6-aaf8-b43d09b70538)

### How AutoRAG optimizes RAG pipeline?

![rag_opt_gif](https://github.com/user-attachments/assets/55bd09cd-8420-4f6d-bc7d-0a66af288317)

## Quick Start

### 1. Set YAML File

First, you need to set the config yaml file for your RAG optimization.

Here is an example of the config yaml file to use `retrieval`, `prompt_maker`, and `generator` nodes.

```yaml
node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_ndcg, retrieval_mrr]
      top_k: 3
      modules:
        - module_type: vectordb
          embedding_model: openai
        - module_type: bm25
        - module_type: hybrid_rrf
          weight_range: (4,80)
- node_line_name: post_retrieve_node_line
  nodes:
    - node_type: prompt_maker
      strategy:
        metrics:
          - metric_name: meteor
          - metric_name: rouge
          - metric_name: sem_score
            embedding_model: openai
      modules:
        - module_type: fstring
          prompt: "Read the passages and answer the given question. \n Question: {query} \n Passage: {retrieved_contents} \n Answer : "
    - node_type: generator
      strategy:
        metrics:
          - metric_name: meteor
          - metric_name: rouge
          - metric_name: sem_score
            embedding_model: openai
      modules:
        - module_type: openai_llm
          llm: gpt-4o-mini
          batch: 16
```

You can get various config yaml files at [here](./sample_config).
We highly recommend using pre-made config yaml files for starter.

If you want to make your own config yaml files, check out the [Config yaml file](#-create-your-own-config-yaml-file)
section.

### 2. Run AutoRAG

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
at [here](https://docs.auto-rag.com/optimization/folder_structure.html).

### 3. Run Dashboard

You can run dashboard to easily see the result.

```bash
autorag dashboard --trial_dir /your/path/to/trial_dir
```

#### sample dashboard

![dashboard](https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/3798827d-31d7-4c4e-a9b1-54340b964e53)

### 4. Deploy your optimal RAG pipeline (for testing)

### 4-1. Run as a CLI

You can use a found optimal RAG pipeline right away with extracted yaml file.

```python
from autorag.deploy import Runner

runner = Runner.from_yaml('your/path/to/pipeline.yaml')
runner.run('your question')
```

### 4-2. Run as an API server

You can run this pipeline as an API server.

Check out API endpoint at [here](deploy/api_endpoint.md).

```python
from autorag.deploy import Runner

runner = Runner.from_yaml('your/path/to/pipeline.yaml')
runner.run_api_server()
```

```bash
autorag run_api --config_path your/path/to/pipeline.yaml --host 0.0.0.0 --port 8000
```

### 4-3. Run as a Web Interface

you can run this pipeline as a web interface.

Check out web interface at [here](deploy/web.md).

```bash
autorag run_web --trial_path your/path/to/trial_path
```

#### sample web interface

<img width="1491" alt="web_interface" src="https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/f6b00353-f6bb-4d8f-8740-1c264c0acbb8">

## 📌 Supporting Data Creation Modules
![Data Creation](https://github.com/user-attachments/assets/97abe6f4-91a9-42e4-b193-11445001e174)

- You can check our all Parsing Modules at [here](https://edai.notion.site/Supporting-Parse-Modules-e0b7579c7c0e4fb2963e408eeccddd75?pvs=4)
- You can check our all Chunk Modules at [here](https://edai.notion.site/Supporting-Chunk-Modules-8db803dba2ec4cd0a8789659106e86a3?pvs=4)

## ❗Supporting RAG Optimization Nodes & modules

![module_1](https://github.com/user-attachments/assets/4a534fd4-800c-4878-bba3-5686d7e20a7c)
![module_2](https://github.com/user-attachments/assets/19c8faa1-e90d-4f4d-99bc-3fa1ded4a178)
![module_3](https://github.com/user-attachments/assets/98a109e7-a4c1-4c74-a9fb-8361c3a12ac3)
![module_4](https://github.com/user-attachments/assets/b355af15-d5ae-4b9e-9869-879c589f466f)

You can check our all supporting Nodes & modules
at [here](https://edai.notion.site/Supporting-Nodes-modules-0ebc7810649f4e41aead472a92976be4?pvs=4)

## ❗Supporting Evaluation Metrics

![Metrics](https://github.com/user-attachments/assets/2d7a342a-6b95-4235-af82-d6537dcd02aa)

You can check our all supporting Evaluation Metrics
at [here](https://edai.notion.site/Supporting-metrics-867d71caefd7401c9264dd91ba406043?pvs=4)

- [Retrieval Metrics](https://edai.notion.site/Retrieval-Metrics-dde3d9fa1d9547cdb8b31b94060d21e7?pvs=4)
- [Retrieval Token Metrics](https://edai.notion.site/Retrieval-Token-Metrics-c3e2d83358e04510a34b80429ebb543f?pvs=4)
- [Generation Metrics](https://edai.notion.site/Retrieval-Token-Metrics-c3e2d83358e04510a34b80429ebb543f?pvs=4)


## ☎️ FaQ

🛣️ [Support plans & Roadmap](https://edai.notion.site/Support-plans-Roadmap-02ca7c97376340c393885855e2d99630?pvs=4)

💻 [Hardware Specs](https://edai.notion.site/Hardware-specs-28cefcf2a26246ffadc91e2f3dc3d61c?pvs=4)

⭐ [Running AutoRAG](https://edai.notion.site/About-running-AutoRAG-44a8058307af42068fc218a073ee480b?pvs=4)

🍯 [Tips/Tricks](https://edai.notion.site/Tips-Tricks-10708a0e36ff461cb8a5d4fb3279ff15?pvs=4)

☎️ [TroubleShooting](https://medium.com/@autorag/autorag-troubleshooting-5cf872b100e3)

---

# ✨ Contributors ✨

Thanks go to these wonderful people:

<a href="https://github.com/Marker-Inc-Korea/AutoRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Marker-Inc-Korea/AutoRAG" />
</a>

# Contribution

We are developing AutoRAG as open-source.

So this project welcomes contributions and suggestions. Feel free to contribute to this project.

Plus, check out our detailed documentation at [here](https://docs.auto-rag.com/index.html).
