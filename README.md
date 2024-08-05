# AutoRAG

RAG AutoML tool for automatically finds an optimal RAG pipeline for your data.

Explore our 📖 [Document](https://docs.auto-rag.com)!!

Plus, join our 📞 [Discord](https://discord.gg/P4DYXfmSAs) Community.

---

### 💪 Colab Tutorial

- [Step 1: Basic of AutoRAG | Optimizing your RAG pipeline](https://colab.research.google.com/drive/19OEQXO_pHN6gnn2WdfPd4hjnS-4GurVd?usp=sharing)

---

### 🚨 YouTube Tutorial

https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/c0d23896-40c0-479f-a17b-aa2ec3183a26

_Muted by default, enable sound for voice-over_

You can see on [YouTube](https://youtu.be/2ojK8xjyXAU?feature=shared)

---

### ☎️ FaQ

🛣️ [Support plans & Roadmap](https://edai.notion.site/Support-plans-Roadmap-02ca7c97376340c393885855e2d99630?pvs=4)

💻 [Hardware Specs](https://edai.notion.site/Hardware-specs-28cefcf2a26246ffadc91e2f3dc3d61c?pvs=4)

⭐ [Running AutoRAG](https://edai.notion.site/About-running-AutoRAG-44a8058307af42068fc218a073ee480b?pvs=4)

🍯 [Tips/Tricks](https://edai.notion.site/Tips-Tricks-10708a0e36ff461cb8a5d4fb3279ff15?pvs=4)

☎️ [TroubleShooting](https://medium.com/@autorag/autorag-troubleshooting-5cf872b100e3)

---

# 😁 Introduction

There are many RAG pipelines and modules out there,
but you don’t know what pipeline is great for “your own data” and "your own use-case."
Making and evaluating all RAG modules is very time-consuming and hard to do.
But without it, you will never know which RAG pipeline is the best for your own use-case.

AutoRAG is a tool for finding optimal RAG pipeline for “your data.”
You can evaluate various RAG modules automatically with your own evaluation data,
and find the best RAG pipeline for your own use-case.

AutoRAG supports a simple way to evaluate many RAG module combinations.
Try now and find the best RAG pipeline for your own use-case.

# 📌AutoRAG Structure

![Node](https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/79dda7ba-e9d8-4552-9e7b-6a5f9edc4c1a)

## ❗Supporting Nodes & modules

![module_1](https://github.com/user-attachments/assets/d49cc7b4-56a6-4b5f-bb3d-2fc5b962996b)
![module_2](https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/4db3d83f-3178-4b81-bad1-b751dcd31a13)
You can check our all supporting Nodes & modules
at [here](https://edai.notion.site/Supporting-Nodes-modules-0ebc7810649f4e41aead472a92976be4?pvs=4)

## ❗Supporting Evaluation Metrics

![Metrics](https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/9f090e3c-d762-41cb-9f1d-cd8d7da137dd)

You can check our all supporting Evaluation Metrics
at [here](https://edai.notion.site/Supporting-metrics-867d71caefd7401c9264dd91ba406043?pvs=4)

- [Retrieval Metrics](https://edai.notion.site/Retrieval-Metrics-dde3d9fa1d9547cdb8b31b94060d21e7?pvs=4)
- [Retrieval Token Metrics](https://edai.notion.site/Retrieval-Token-Metrics-c3e2d83358e04510a34b80429ebb543f?pvs=4)
- [Generation Metrics](https://edai.notion.site/Retrieval-Token-Metrics-c3e2d83358e04510a34b80429ebb543f?pvs=4)

# ⚡ Quick Install

We recommend using Python version 3.9 or higher for AutoRAG.

```bash
pip install AutoRAG
```

# ⚡ QuickStart

### 1. Prepare your evaluation data

For evaluation, you need to prepare just three files.

1. **QA** dataset file (qa.parquet)
2. **Corpus** dataset file (corpus.parquet)
3. **Config yaml file** (config.yaml)

There is a template for your evaluation data for using AutoRAG.

- Check out how to make evaluation data
  at [here](https://docs.auto-rag.com/data_creation/tutorial.html).
- Check out the evaluation data rule
  at [here](https://docs.auto-rag.com/data_creation/data_format.html).
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
at [here](https://docs.auto-rag.com/optimization/folder_structure.html).

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
Check out API endpoint at [here](https://docs.auto-rag.com/deploy/api_endpoint.html).

```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run_api_server()
```

You can run api server with CLI command.

```bash
autorag run_api --config_path your/path/to/pipeline.yaml --host 0.0.0.0 --port 8000
```

### 4. Run Dashboard

You can run dashboard to easily see the result.

```bash
autorag dashboard --trial_dir /your/path/to/trial_dir
```

- sample dashboard:

![dashboard](https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/3798827d-31d7-4c4e-a9b1-54340b964e53)

### 5. Share your RAG pipeline

You can use your RAG pipeline from extracted pipeline yaml file.
This extracted pipeline is great for sharing your RAG pipeline to others.

You must run this at project folder, which contains datas in data folder, and ingested corpus for retrieval at resources
folder.

```python
from autorag.deploy import extract_best_config

pipeline_dict = extract_best_config(trial_path='your/path/to/trial_folder', output_path='your/path/to/pipeline.yaml')
```

### 6. Deploy your optimal RAG pipeline (for testing)

### 6-1. Run as a CLI

You can use a found optimal RAG pipeline right away with extracted yaml file.

```python
from autorag.deploy import Runner

runner = Runner.from_yaml('your/path/to/pipeline.yaml')
runner.run('your question')
```

### 6-2. Run as an API server

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

### 6-3. Run as a Web Interface

you can run this pipeline as a web interface.

Check out web interface at [here](deploy/web.md).

```bash
autorag run_web --trial_path your/path/to/trial_path
```

- sample web interface:

<img width="1491" alt="web_interface" src="https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/f6b00353-f6bb-4d8f-8740-1c264c0acbb8">

# ⭐ Star History ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Marker-Inc-Korea/AutoRAG&type=Date)](https://star-history.com/#Marker-Inc-Korea/AutoRAG&Date)

# ✨ Contributors ✨

Thanks go to these wonderful people:

<a href="https://github.com/Marker-Inc-Korea/AutoRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Marker-Inc-Korea/AutoRAG" />
</a>

# Contribution

We are developing AutoRAG as open-source.

So this project welcomes contributions and suggestions. Feel free to contribute to this project.

Plus, check out our detailed documentation at [here](https://docs.auto-rag.com/index.html).
