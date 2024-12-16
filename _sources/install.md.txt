---
myst:
   html_meta:
      title: AutoRAG - Installation and Setup
      description: Learn how to install AutoRAG
      keywords: AutoRAG,RAG,AutoRAG install
---
# Installation and Setup

To install AutoRAG, you can use pip:

```bash
pip install AutoRAG
```

Plus, it is recommended to install PyOpenSSL and nltk libraries for full features.

```bash
pip install --upgrade pyOpenSSL
pip install nltk
python3 -c "import nltk; nltk.download('punkt_tab')"
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

```{admonition} Trouble with installation?
Do you have any trouble with installation?
First, you can check out the [troubleshooting](troubleshooting.md) page.
```

### Note for Windows Users
AutoRAG is not fully supported on Windows yet. There are several constraints for Windows users.

1. TART, UPR, and MonoT5 passage rerankers does not support Windows.
2. Parsing might be not working properly in the Windows environment.
3. Cannot use FlagEmbedding passage reranker with `batch` setting with 1. The default batch is 64.

Due to the constraints, we recommend using Docker images for running AutoRAG on Windows.

Plus, you MAKE SURE UPGRADE UP TO v0.3.1 for Windows users.

### Installation for Local Models 🏠

For using local models, you need to install some additional dependencies.

```bash
pip install "AutoRAG[gpu]"
```

### Installation for Parsing 🌲

For parsing you need to install some local packages like [libmagic](https://man7.org/linux/man-pages/man3/libmagic.3.html),
[tesseract](https://github.com/tesseract-ocr/tesseract), and [poppler](https://poppler.freedesktop.org/).
The installation method depends upon your OS.

After installing this, you can install AutoRAG with parsing like below.

```bash
pip install "AutoRAG[parse]"
```

### Installation for Korean 🇰🇷

You can install optional dependencies for the Korean language.

```bash
pip install "AutoRAG[ko]"
```

And after that, you have to install **jdk 17** for using `konlpy`.
Plus, remember to set environment PATH for jdk.
(JAVA_HOME and PATH)

The instruction for Mac users is [here](https://velog.io/@yoonsy/M1%EC%B9%A9-Mac%EC%97%90-konlpy-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0).

### Installation for Japanese 🇯🇵

```bash
pip install "AutoRAG[ja]"
```

## Setup OPENAI API KEY
To use LLM and embedding models, it is common to use OpenAI models.
If you want to use other models, check out [here](local_model.md)

You need to set OPENAI_API_KEY environment variable.
You can get your API key at [here](https://platform.openai.com/account/api-keys).

```bash
export OPENAI_API_KEY="sk-...your-api-key..."
```

Or, as an alternative, you can set env variable using the `.env` file.

```bash
pip install python-dotenv
```

Then, make an.env file at your root folder like below.
```dotenv
OPENAI_API_KEY=sk-...your-api-key...
```

And when you try to run AutoRAG, you can use below code to load `.env` file.

```python
from dotenv import load_dotenv

load_dotenv()
```

And you are ready to use AutoRAG!


## Build from source

If you want to build AutoRAG from source, the first step is to clone the AutoRAG repository.

```bash
git clone https://github.com/Marker-Inc-Korea/AutoRAG.git
```

And install AutoRAG to editable.
```bash
cd AutoRAG
pip install -e .
```

And then, for testing and documentation build, you need to install some additional packages.

```bash
pip install -r tests/requirements.txt
pip install -r docs/requirements.txt
```

For testing, you have to set the environment variable at pytest.ini.
Make a new `pytest.ini` file at the root of the project and write below.

```ini
[pytest]
env =
    OPENAI_API_KEY=sk-...your-api-key...

log_cli=true
log_cli_level=INFO
```

After that, you can run tests with pytest.

```bash
python -m pytest -n auto
```

After this, please check out our documentation for contributors.
We are writing this documentation for contributors, so please wait for a while.


## Run AutoRAG with 🐳 Docker

Tip: If you want to build an image for a gpu version, you can use `autoraghq/autorag:gpu` or `autoraghq/autorag:gpu-parsing`

To run AutoRAG using Docker, follow these steps:

### 1. Build the Docker Image

```bash
docker build --target production -t autorag:prod .
```

This command will build the production-ready Docker image, using only the `production` stage defined in the `Dockerfile`.

### 2. Run the Docker Container

Run the container with the following command:

```bash
docker run --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/sample_config:/usr/src/app/sample_config \
  -v $(pwd)/projects:/usr/src/app/projects \
  autoraghq/autorag:api evaluate \
  --config /usr/src/app/sample_config/rag/simple/simple_openai.yaml \
  --qa_data_path /usr/src/app/projects/test01/qa_validation.parquet \
  --corpus_data_path /usr/src/app/projects/test01/corpus.parquet \
  --project_dir /usr/src/app/projects/test01
```

#### Explanation:
- **`-v ~/.cache/huggingface:/root/.cache/huggingface`**: Mounts the host's Hugging Face cache to the container, allowing it to access pre-downloaded models.
- **`-v $(pwd)/sample_config:/usr/src/app/sample_config`**: Mounts the local `sample_config` directory to the container.
- **`-v $(pwd)/projects:/usr/src/app/projects`**: Mounts the local `projects` directory to the container.
- **`autoraghq/autorag:all evaluate`**: Executes the `evaluate` command inside the `autoraghq/autorag:all` container.
- **`--config`, `--qa_data_path`, `--corpus_data_path`, `--project_dir`**: Specifies paths to the configuration file, QA dataset, corpus data, and project directory.

### 3. Using a Custom Cache Directory with `HF_HOME`

Alternatively, you can mount the Hugging Face cache to a custom location inside the container and set the `HF_HOME` environment variable:

```bash
docker run --rm -it \
  -v ~/.cache/huggingface:/cache/huggingface \
  -v $(pwd)/sample_config:/usr/src/app/sample_config \
  -v $(pwd)/projects:/usr/src/app/projects \
  -e HF_HOME=/cache/huggingface \
  autoraghq/autorag:api evaluate \
  --config /usr/src/app/sample_config/rag/simple/simple_openai.yaml \
  --qa_data_path /usr/src/app/projects/test01/qa_validation.parquet \
  --corpus_data_path /usr/src/app/projects/test01/corpus.parquet \
  --project_dir /usr/src/app/projects/test01
```

#### Explanation:
- **`-v ~/.cache/huggingface:/cache/huggingface`**: Mounts the host's Hugging Face cache to `/cache/huggingface` inside the container.
- **`-e HF_HOME=/cache/huggingface`**: Sets the `HF_HOME` environment variable to point to the mounted cache directory.

### 5. Debugging and Manual Access

To manually access the container for debugging or testing, start a Bash shell:

```bash
docker run --rm -it --entrypoint /bin/bash autoraghq/autorag:api
```

This command allows you to explore the container’s filesystem, run commands manually, or inspect logs for troubleshooting.

### 6. Use gpu version

To use the gpu version, you must install CUDA and cuDNN in your host system.
It built on the cuda 11.8 version and pytorch docker image.

```bash
docker run --rm -it \
  -v ~/.cache/huggingface:/cache/huggingface \
  -v $(pwd)/sample_config:/usr/src/app/sample_config \
  -v $(pwd)/projects:/usr/src/app/projects \
  -e HF_HOME=/cache/huggingface \
  --gpus all \ # Be sure to add this line
  autoraghq/autorag:gpu evaluate \
  --config /usr/src/app/sample_config/rag/simple/simple_openai.yaml \
  --qa_data_path /usr/src/app/projects/test01/qa_validation.parquet \
  --corpus_data_path /usr/src/app/projects/test01/corpus.parquet \
  --project_dir /usr/src/app/projects/test01
```

## Additional Notes

- Ensure that the necessary directories (`sample_config` and `projects`) are present in the host system.
- If running in a CI/CD pipeline, consider using environment variables or `.env` files to manage API keys and paths dynamically.
