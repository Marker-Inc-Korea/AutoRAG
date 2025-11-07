# AutoRAG - AI-Powered RAG Pipeline Optimization

AutoRAG is a Python package for automatically finding optimal RAG (Retrieval-Augmented Generation) pipelines for your data using AutoML techniques. The system evaluates various RAG module combinations to find the best configuration for your specific use case.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Prerequisites and System Setup
- **CRITICAL**: Use Python 3.10 or higher (Python 3.12 recommended)
- Install UV package manager for faster dependency management: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For systems without UV, fallback to pip installation
- Install system dependencies first (see Platform Dependencies section)

### Platform Dependencies
Install these system packages before building AutoRAG:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc poppler-utils tesseract-ocr libssl-dev
```

**Java (required for some features):**
```bash
# Install Java 17 (required)
sudo apt-get install openjdk-17-jdk
```

### Bootstrap, Build, and Test - COMPLETE WORKFLOW

**TIMING EXPECTATIONS:**
- **NEVER CANCEL builds or tests** - they can take 15-45+ minutes
- Initial dependency installation: 15-30 minutes
- Full test suite: 15-20 minutes
- Documentation build: 5-10 minutes
- **ALWAYS set timeouts to 60+ minutes for builds and 30+ minutes for tests**

**1. Clone and Environment Setup:**
```bash
git clone https://github.com/Marker-Inc-Korea/AutoRAG.git
cd AutoRAG/autorag
```

**2. Create Virtual Environment and Install (Primary Method - UV):**
```bash
# Install UV if not available
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # Reload shell

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install AutoRAG with all extras - TAKES 15-30 MINUTES, NEVER CANCEL
uv pip install -r pyproject.toml --all-extras -e . --timeout 3600

# Install development dependencies - TAKES 5-10 MINUTES
uv pip install --group dev --timeout 1800
```

**3. Fallback Installation Method (Pip):**
```bash
# If UV fails, use pip
python3 -m venv venv
source venv/bin/activate

# Install AutoRAG - TAKES 20-45 MINUTES, NEVER CANCEL
pip install -e '.[all]' --timeout 3600

# Install dev dependencies
pip install ruff pre-commit pytest pytest-env pytest-xdist pytest-asyncio pytest-mock aioresponses asyncstdlib
```

**4. Additional Required Dependencies:**
```bash
# Upgrade SSL and install NLTK - REQUIRED
pip install --upgrade pyOpenSSL
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

**5. Setup Testing Environment:**
Create `pytest.ini` file in the repository root:
```bash
cat > pytest.ini << 'EOF'
[pytest]
env =
    OPENAI_API_KEY=sk-your-api-key-here

log_cli=true
log_cli_level=INFO
EOF
```

**6. Run Tests - CRITICAL TIMING:**
```bash
# Delete conflicting test packages first
python tests/delete_tests.py

# Run tests - TAKES 15-20 MINUTES, NEVER CANCEL
python -m pytest -o log_cli=true --log-cli-level=INFO -n auto tests/autorag --timeout 1800
```

**7. Setup Pre-commit Hooks:**
```bash
pre-commit install
```

### Daily Development Commands

**Run the CLI (after installation):**
```bash
# Check available commands
autorag --help

# Main commands:
autorag evaluate --config config.yaml --qa_data_path qa.parquet --corpus_data_path corpus.parquet --project_dir ./project
autorag validate --config config.yaml --qa_data_path qa.parquet --corpus_data_path corpus.parquet
autorag run_api --trial_path ./trial --host 0.0.0.0 --port 8000
autorag dashboard --trial_dir ./trial --port 7690
autorag run_web --trial_path ./trial --host 0.0.0.0 --port 8501
```

**Build Documentation:**
```bash
cd docs/
# Install docs dependencies - TAKES 3-5 MINUTES
pip install -r requirements.txt

# Build documentation - TAKES 5-10 MINUTES, NEVER CANCEL
sphinx-build -b html source build/html --timeout 600
```

**Code Quality and Linting:**
```bash
# Format and lint code (run before committing)
ruff check --fix
ruff format

# These commands are FAST (< 30 seconds each)
```

## Validation and Testing

### Manual Validation Scenarios
**ALWAYS run these validation scenarios after making changes:**

1. **CLI Basic Validation:**
   ```bash
   # Test CLI help works
   autorag --help

   # Test configuration validation (use actual existing files)
   autorag validate --config autorag/sample_config/rag/full.yaml \
     --qa_data_path projects/tutorial_1/qa/qa_8c42b9e6-490d-4971-bb9e-705b36b7a3a2.parquet \
     --corpus_data_path projects/tutorial_1/parse/parse_8c42b9e6-490d-4971-bb9e-705b36b7a3a2/0.parquet
   ```

2. **API Server Test:**
   ```bash
   # Start API server (runs in background)
   autorag run_api --trial_path ./projects/tutorial_1 --host 0.0.0.0 --port 8000 &

   # Test API endpoint
   curl http://localhost:8000/health

   # Stop server
   pkill -f "autorag run_api"
   ```

3. **Dashboard Test:**
   ```bash
   # Start dashboard (runs in background)
   autorag dashboard --trial_dir ./projects/tutorial_1 --port 7690 &

   # Verify it's running
   curl http://localhost:7690

   # Stop dashboard
   pkill -f "autorag dashboard"
   ```

4. **Full Evaluation Pipeline Test:**
   ```bash
   # WARNING: This test requires OpenAI API key and may cost ~$0.30
   # Use sample data from projects/tutorial_1/ (actual files have UUIDs in names)
   autorag evaluate \
     --config projects/tutorial_1/configs/parse_config_8c42b9e6-490d-4971-bb9e-705b36b7a3a2.yaml \
     --qa_data_path projects/tutorial_1/qa/qa_8c42b9e6-490d-4971-bb9e-705b36b7a3a2.parquet \
     --corpus_data_path projects/tutorial_1/parse/parse_8c42b9e6-490d-4971-bb9e-705b36b7a3a2/0.parquet \
     --project_dir projects/tutorial_1/
   ```

### Testing Warnings and Notes
- **OpenAI API Key Required**: Many tests require `OPENAI_API_KEY` environment variable
- **GitHub Actions**: Tests may fail in CI due to missing API keys - this is expected
- **GPU Tests**: Some tests require CUDA - use `@pytest.mark.skipif(is_github_action())` to skip in CI
- **Test Data**: Tests must not leave extra files behind
- **Mock Usage**: Heavy GPU models and external API calls must use mocks and pytest fixtures

## Common Issues and Troubleshooting

### Installation Issues
- **Network timeouts**: Use `--timeout 3600` flag with pip/uv commands
- **Missing system packages**: Install platform dependencies first
- **Python version**: Ensure Python 3.10+ is being used
- **Virtual environment**: Always use a virtual environment to avoid conflicts

### Build Failures
- **Java not found**: Install openjdk-17-jdk
- **SSL errors**: Run `pip install --upgrade pyOpenSSL`
- **NLTK missing**: Download required NLTK data as shown above
- **Permission errors**: Use virtual environment, avoid system-wide installs

### Test Failures
- **API key missing**: Set OPENAI_API_KEY in pytest.ini or environment
- **Import errors**: Run `python tests/delete_tests.py` to clean conflicting packages
- **Timeout errors**: Tests can take 15-20 minutes, never cancel early

## Project Structure and Key Locations

### Repository Layout
```
/
├── .github/                 # GitHub workflows and CI/CD
├── autorag/                 # Main Python package directory
│   ├── autorag/            # Core AutoRAG package
│   │   ├── cli.py          # Command-line interface
│   │   ├── evaluator.py    # Pipeline evaluation logic
│   │   ├── dashboard.py    # Web dashboard
│   │   ├── nodes/          # RAG pipeline nodes
│   │   ├── vectordb/       # Vector database integrations
│   │   └── web.py          # Web interface
│   ├── sample_config/      # Sample configuration files
│   │   ├── rag/            # RAG pipeline configs
│   │   ├── chunk/          # Chunking configs
│   │   └── parse/          # Parsing configs
│   └── pyproject.toml      # Main project configuration
├── tests/                  # Test suite
├── docs/                   # Sphinx documentation
├── projects/               # Sample projects and datasets
└── api/                    # API server implementation
```

### Key Files to Check After Changes
- **Configuration Changes**: Always validate with `autorag validate`
- **CLI Changes**: Test with `autorag --help` and specific commands
- **API Changes**: Check `autorag/autorag/deploy/api.py` and test endpoints
- **Node Changes**: Validate in `autorag/autorag/nodes/` and run related tests
- **Config Changes**: Update sample configs in `autorag/sample_config/`

### Configuration Files
- **Sample Configs**: `autorag/sample_config/rag/full.yaml` - comprehensive example
- **Docker Config**: `autorag/Dockerfile.base` - multi-stage build configuration
- **Test Config**: `projects/tutorial_1/` - working example with test data
- **Dependencies**: `autorag/pyproject.toml` - all package dependencies and extras

## Docker Usage (Alternative Deployment)

**Pre-built Images Available:**
- `autoraghq/autorag:api-latest` - API server
- `autoraghq/autorag:all` - Full installation
- `autoraghq/autorag:gpu` - GPU-enabled version

**Quick Docker Test:**
```bash
# Download sample data first, then run evaluation
docker run --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/projects:/usr/src/app/projects \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  autoraghq/autorag:api-latest evaluate \
  --config /usr/src/app/projects/tutorial_1/configs/parse_config_8c42b9e6-490d-4971-bb9e-705b36b7a3a2.yaml \
  --qa_data_path /usr/src/app/projects/tutorial_1/qa/qa_8c42b9e6-490d-4971-bb9e-705b36b7a3a2.parquet \
  --corpus_data_path /usr/src/app/projects/tutorial_1/parse/parse_8c42b9e6-490d-4971-bb9e-705b36b7a3a2/0.parquet \
  --project_dir /usr/src/app/projects/tutorial_1/
```

## Development Workflow

### Before Making Changes
1. **ALWAYS** run the bootstrap and test sequence above
2. Create feature branch: `git checkout -b feature/your-feature`
3. Set up pre-commit hooks: `pre-commit install`

### Making Changes
1. **Make minimal changes** - change as few lines as possible
2. **Follow existing patterns** in the codebase
3. **Add tests** for new functionality in `tests/autorag/`
4. **Update configs** if adding new nodes or features

### Before Committing
1. **Run formatting**: `ruff check --fix && ruff format`
2. **Run affected tests**: `python -m pytest tests/autorag/path/to/relevant/tests`
3. **Validate CLI still works**: `autorag --help`
4. **Test your changes manually** using the validation scenarios above

### Submitting Changes
1. **Ensure all linting passes**: CI will fail if ruff checks fail
2. **Include test coverage**: Add unit tests for new features
3. **Document breaking changes**: Update relevant config samples
4. **Test with sample data**: Ensure tutorial examples still work

## Timing Expectations Summary

| Operation | Expected Time | Timeout Setting | Notes |
|-----------|---------------|-----------------|--------|
| Initial install | 15-30 min | 60+ min | NEVER CANCEL |
| Full test suite | 15-20 min | 30+ min | NEVER CANCEL |
| Documentation build | 5-10 min | 15+ min | Usually fast |
| Code formatting | < 30 sec | 2 min | Very fast |
| Single test file | 1-5 min | 10 min | Varies by test |
| API evaluation | 5-15 min | 20+ min | Depends on data size |

**CRITICAL**: Always wait for builds and tests to complete. The CI pipeline expects these timings and builds may appear to hang but are actually processing large dependency trees.
