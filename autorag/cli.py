import logging
import os

import click

from autorag.deploy import Runner
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


cli.add_command(evaluate, 'evaluate')
cli.add_command(run_api, 'run_api')

if __name__ == '__main__':
    cli()
