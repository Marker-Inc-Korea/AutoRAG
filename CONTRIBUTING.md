# Getting Started

Thank you so much for your consideration of contributing to AutoRAG!
Okay Let's get started!

## Quickstart

1. Fork the repo and clone your fork
2. Create a new virtual environment and activate it
3. Install the dependencies and package (`pip install -e '.[all]'`)
4. Install docs and test dependencies (`pip install -r docs/requirements.txt` and `pip install -r tests/requirements.txt`)

Also, join our Discord at [here](https://discord.gg/P4DYXfmSAs) and discuss about your contribution with the community.

## Contribution Guidelines

### What should I work on?

1. Add new modules or nodes to the AutoRAG.
2. Fix bugs
3. Add new feature to the AutoRAG
4. Add new sample config YAML files
5. Add new sample dataset
6. Improve code quality & documentation
7. Report a bug or suggest new feature

It can be great you search 'good first issue' tag on github issue tab.

# Development Guidelines

## Repo Structure

The `AutoRAG` repo have a standard python library structure.

You can find the core part of the library in `autorag` folder.
You can find the test codes of the `AutoRAG` at the `tests` folder.
Plus, you can find the [AutoRAG docs](https://docs.auto-rag.com) at the `docs` folder.

For resources, you can find sample config YAML files at `sample_config` folder.
Plus, you can find sample dataset download script from `sample_dataset` folder.

## Setting up environment

As a python package, AutoRAG tested primarily in Python versions >= 3.9.
Here's a guide to set environment.

1. Fork [AutoRAG Github repo](https://github.com/Marker-Inc-Korea/AutoRAG) and clone it to your local machine. (New to GitHub / git? Here's [how](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).)
2. Open your cloned repository to your IDE or open your project at the terminal through `cd`.
3. Install pre-commit hooks by running `pre-commit install` in the terminal.
4. Make a new virtual environment (Highly Recommended)
5. Install AutoRAG as development version. `pip install -e '.[all]'`
6. Install test dependencies `pip install -r tests/requirements.txt`
7. Write `pytest.ini` file and add env variable for running tests. (We do not need OPENAI_API_KEY at GitHub actions)
```ini
[pytest]
env =
    OPENAI_API_KEY=sk-xxxx

log_cli=true
log_cli_level=INFO
```
8. Install docs dependencies `pip install -r docs/requirements.txt`

## Validating your change

Let's make sure to `format/lint` our change.
For bigger changes, let's also make sure to `test` our change.

### Formatting/Linting

We use `ruff` to linting and formatting our code.
If you have installed pre-commit hooks in this repo, they should have taken care of the formatting and linting automatically.
(Use `pre-commit install`)

If you want to do it manually, you can lint and format with ruff.

```bash
ruff check --fix
ruff format
```

### Testing

If you modified or added new code logic, you have to create tests for them, too.
- In almost all cases, add unit tests.
- If your new features uses heavy GPU models or outside API connections, you must use mock and pytest fixtures.
- The added test case must not leave any extra files.
- You must cover all cases of features or adjust that you made.
- If you don't know well about `pytest`, `pytest-env` and pytest fixtures and mock, please learn about it.
- Lastly, if the test you made must need CUDA-enabled device, you can mark the test with `@pytest.mark.skipif(is_github_action())` so we can skip it on github actions.

Plus, your test will run as github actions when you try to merge it to the main branch.

#### Warning about OPENAI_API_KEY

When executing test at github actions, it will be failed due to the OpenAI API key.
Don't worry, we will run the test locally and merge your code if there is no problem.

## Making a Pull Request

You can make a pull request to the main branch of the AutoRAG repo.
At the fork repository, you will make a PR to AutoRAG repo.
You can assign the issue with `close #issue_number` to automatically close the issue when the PR is merged.

After that, please assign `vkehfdl1` or `bwook00` as a reviewer.
For merge, you will have to get at least one approval from the reviewer.
