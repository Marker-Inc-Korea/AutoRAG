import importlib.resources
import logging
import os
import pathlib
import subprocess
from pathlib import Path
from typing import Optional

import click

from autorag.deploy import Runner
from autorag.deploy import extract_best_config as original_extract_best_config
from autorag.evaluator import Evaluator

logger = logging.getLogger("AutoRAG")


@click.group()
def cli():
    pass


@click.command()
@click.option('--config', '-c', help='Path to config yaml file. Must be yaml or yml file.', type=str)
@click.option('--qa_data_path', help='Path to QA dataset. Must be parquet file.', type=str)
@click.option('--corpus_data_path', help='Path to corpus dataset. Must be parquet file.', type=str)
@click.option('--project_dir', help='Path to project directory.', type=str, default=None)
def evaluate(config, qa_data_path, corpus_data_path, project_dir):
    if not config.endswith('.yaml') and not config.endswith('.yml'):
        raise ValueError(f"Config file {config} is not a parquet file.")
    if not os.path.exists(config):
        raise ValueError(f"Config file {config} does not exist.")
    evaluator = Evaluator(qa_data_path, corpus_data_path, project_dir=project_dir)
    evaluator.start_trial(config)
    logger.info('Evaluation complete.')


@click.command()
@click.option('--config_path', type=str, help='Path to extracted config yaml file.')
@click.option('--host', type=str, default='0.0.0.0', help='Host address')
@click.option('--port', type=int, default=8000, help='Port number')
@click.option('--project_dir', help='Path to project directory.', type=str, default=None)
def run_api(config_path, host, port, project_dir):
    runner = Runner.from_yaml(config_path, project_dir=project_dir)
    logger.info(f"Running API server at {host}:{port}...")
    runner.run_api_server(host, port)


@click.command()
@click.option('--yaml_path', type=click.Path(path_type=Path), help='Path to the YAML file.')
@click.option('--project_dir', type=click.Path(path_type=Path), help='Path to the project directory.')
@click.option('--trial_path', type=click.Path(path_type=Path), help='Path to the trial directory.')
def run_web(yaml_path: Optional[str], project_dir: Optional[str], trial_path: Optional[str]):
    try:
        with importlib.resources.path('autorag', 'web.py') as web_path:
            web_py_path = str(web_path)
    except ImportError:
        raise ImportError("Could not locate the web.py file within the autorag package."
                          " Please ensure that autorag is correctly installed.")

    if not yaml_path and not trial_path:
        raise ValueError("yaml_path or trial_path must be given.")
    elif yaml_path and trial_path:
        raise ValueError("yaml_path and trial_path cannot be given at the same time.")
    elif yaml_path and not project_dir:
        subprocess.run(['streamlit', 'run', web_py_path, '--', '--yaml_path', yaml_path])
    elif yaml_path and project_dir:
        subprocess.run(['streamlit', 'run', web_py_path, '--', '--yaml_path', yaml_path, '--project_dir', project_dir])
    elif trial_path:
        subprocess.run(['streamlit', 'run', web_py_path, '--', '--trial_path', trial_path])


@click.command()
@click.option('--trial_path', type=click.Path(), help='Path to the trial directory.')
@click.option('--output_path', type=click.Path(), help='Path to the output directory.'
                                                       ' Must be .yaml or .yml file.')
def extract_best_config(trial_path: str, output_path: str):
    original_extract_best_config(trial_path, output_path)


@click.command()
@click.option('--trial_path', help='Path to trial directory.', type=str)
def restart_evaluate(trial_path):
    if not os.path.exists(trial_path):
        raise ValueError(f"trial_path {trial_path} does not exist.")
    project_dir = pathlib.PurePath(trial_path).parent
    qa_data_path = os.path.join(project_dir, 'data', 'qa.parquet')
    corpus_data_path = os.path.join(project_dir, 'data', 'corpus.parquet')
    evaluator = Evaluator(qa_data_path, corpus_data_path, project_dir)
    evaluator.restart_trial(trial_path)
    logger.info('Evaluation complete.')


cli.add_command(evaluate, 'evaluate')
cli.add_command(run_api, 'run_api')
cli.add_command(run_web, 'run_web')
cli.add_command(extract_best_config, 'extract_best_config')
cli.add_command(restart_evaluate, 'restart_evaluate')

if __name__ == '__main__':
    cli()
