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

### Installation for Local Models 🏠

For using local models, you need to install some additional dependencies.

```bash
pip install "AutoRAG[gpu]"
```

### Installation for vLLM ⚡

If you plan to use vLLM (as a generator or embedding backend), install the `vllm` optional dependencies.

```bash
# pip
pip install "AutoRAG[vllm]"

# uv (recommended)
uv pip install "AutoRAG[vllm]"
```

When developing from source, you can also install everything at once:

```bash
uv pip install -e .[all]
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

## Additional Notes

- Ensure that the necessary directories (`sample_config` and `projects`) are present in the host system.
- If running in a CI/CD pipeline, consider using environment variables or `.env` files to manage API keys and paths dynamically.
