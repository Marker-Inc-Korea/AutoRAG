import logging
import os
import uuid
from typing import Optional, Dict

import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from autorag.support import get_support_modules
from autorag.utils.util import load_summary_file

logger = logging.getLogger("AutoRAG")


def summary_df_to_yaml(summary_df: pd.DataFrame) -> Dict:
    """
    Convert trial summary dataframe to config yaml file.

    :param summary_df:
    :return: Dictionary of config yaml file.
        You can save this dictionary to yaml file.
    """
    # summary_df columns : 'node_line_name', 'node_type', 'best_module_filename',
    #                      'best_module_name', 'best_module_params', 'best_execution_time'
    grouped = summary_df.groupby('node_line_name')

    node_lines = [
        {
            'node_line_name': node_line_name,
            'nodes': [
                {
                    'node_type': row['node_type'],
                    'modules': [{
                        'module_type': row['best_module_name'],
                        **row['best_module_params']
                    }]
                }
                for _, row in node_line.iterrows()
            ]
        }
        for node_line_name, node_line in grouped
    ]
    return {'node_lines': node_lines}


def extract_best_config(trial_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Extract the optimal pipeline from evaluated trial.

    :param trial_path: The path to the trial directory that you want to extract the pipeline from.
        Must already be evaluated.
    :param output_path: Output path that pipeline yaml file will be saved.
        Must be .yaml or .yml file.
        If None, it does not save yaml file and just return dict values.
        Default is None.
    :return: The dictionary of the extracted pipeline.
    """
    summary_path = os.path.join(trial_path, 'summary.parquet')
    if not os.path.exists(summary_path):
        raise ValueError(f"summary.parquet does not exist in {trial_path}.")
    trial_summary_df = load_summary_file(summary_path, dict_columns=['best_module_params'])
    yaml_dict = summary_df_to_yaml(trial_summary_df)
    if output_path is not None:
        with open(output_path, 'w') as f:
            yaml.dump(yaml_dict, f)
    return yaml_dict


class Runner:
    def __init__(self, config: Dict):
        self.config = config
        self.project_dir = os.getcwd()
        self.app = FastAPI()
        self.__add_api_route()

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load Runner from yaml file.
        Must be extracted yaml file from evaluated trial using extract_best_config method.

        :param yaml_path: The path of the yaml file.
        :return: Initialized Runner.
        """
        with open(yaml_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                logger.error(exc)
                raise exc
        return cls(config)

    @classmethod
    def from_trial_folder(cls, trial_path: str):
        """
        Load Runner from evaluated trial folder.
        Must already be evaluated using Evaluator class.

        :param trial_path: The path of the trial folder.
        :return: Initialized Runner.
        """
        config = extract_best_config(trial_path)
        return cls(config)

    def run(self, query: str, result_column: str = "answer"):
        """
        Run the pipeline with query.
        The loaded pipeline must start with a single query,
        so the first module of the pipeline must be `query_expansion` or `retrieval` module.

        :param query: The query of the user.
        :param result_column: The result column name for the answer.
            Default is `answer`, which is the output of the `generation` and `answer_filter` module.
        :return: The result of the pipeline.
        """
        node_lines = self.config['node_lines']
        previous_result = pd.DataFrame({
            'qid': str(uuid.uuid4()),
            'query': [query],
            'retrieval_gt': [[]],
            'generation_gt': [''],
        })  # pseudo qa data for execution
        for node_line in node_lines:
            for node in node_line['nodes']:
                if len(node['modules']) != 1:
                    raise ValueError("The number of modules in a node must be 1 for using runner."
                                     "Please use extract_best_config method for extracting yaml file from evaluated trial.")
                module = node['modules'][0]
                module_type = module.pop('module_type')
                module_params = module
                previous_result = get_support_modules(module_type)(
                    project_dir=self.project_dir,
                    previous_result=previous_result,
                    **module_params
                )
        return previous_result[result_column].tolist()[0]

    def __add_api_route(self):

        @self.app.post("/run")
        async def run_pipeline(runner_input: RunnerInput):
            query = runner_input.query
            result_column = runner_input.result_column
            result = self.run(query, result_column)
            return {result_column: result}

    def run_api_server(self, host: str = '0.0.0.0', port: int = 8000, **kwargs):
        """
        Run the pipeline as api server.
        You can send POST request to `http://host:port/run` with json body like below:

        .. Code:: json

            {
                "Query": "your query",
                "result_column": "answer"
            }

        And it returns json response like below:

        .. Code:: json
            {
                "answer": "your answer"
            }

        :param host: The host of the api server.
        :param port: The port of the api server.
        :param kwargs: Other arguments for uvicorn.run.
        """
        logger.info(f"Run api server at {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


class RunnerInput(BaseModel):
    query: str
    result_column: str = "answer"
