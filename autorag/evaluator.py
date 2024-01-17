import json
import os
import shutil
from datetime import datetime
from typing import List, Dict
import logging

import pandas as pd
import yaml

from autorag.node_line import run_node_line
from autorag.nodes.retrieval.bm25 import bm25_ingest
from autorag.schema import Node
from autorag.schema.node import module_type_exists
from autorag.utils import cast_qa_dataset, cast_corpus_dataset

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, qa_data_path: str, corpus_data_path: str):
        # validate data paths
        if not os.path.exists(qa_data_path):
            raise ValueError(f"QA data path {qa_data_path} does not exist.")
        if not os.path.exists(corpus_data_path):
            raise ValueError(f"Corpus data path {corpus_data_path} does not exist.")
        if not qa_data_path.endswith('.parquet'):
            raise ValueError(f"QA data path {qa_data_path} is not a parquet file.")
        if not corpus_data_path.endswith('.parquet'):
            raise ValueError(f"Corpus data path {corpus_data_path} is not a parquet file.")
        self.qa_data = pd.read_parquet(qa_data_path)
        self.corpus_data = pd.read_parquet(corpus_data_path)
        self.qa_data = cast_qa_dataset(self.qa_data)
        self.corpus_data = cast_corpus_dataset(self.corpus_data)

        # copy dataset to project directory
        if not os.path.exists(os.path.join(os.getcwd(), 'data')):
            os.makedirs(os.path.join(os.getcwd(), 'data'))
        qa_path_in_project = os.path.join(os.getcwd(), 'data', 'qa.parquet')
        if not os.path.exists(qa_path_in_project):
            shutil.copy(qa_data_path, qa_path_in_project)
        corpus_path_in_project = os.path.join(os.getcwd(), 'data', 'corpus.parquet')
        if not os.path.exists(corpus_path_in_project):
            shutil.copy(corpus_data_path, corpus_path_in_project)

        self.project_dir = os.getcwd()

    def start_trial(self, yaml_path: str):
        trial_name = self.__get_new_trial_name()
        self.__make_trial_dir(trial_name)

        node_lines = self._load_node_lines(yaml_path)
        self.__ingest(node_lines)

        for i, (node_line_name, node_line) in enumerate(node_lines.items()):
            logger.info(f'Running node line {node_line_name}...')
            node_line_dir = os.path.join(self.project_dir, trial_name, node_line_name)
            os.makedirs(node_line_dir, exist_ok=False)
            if i == 0:
                previous_result = self.qa_data
            previous_result = run_node_line(node_line, node_line_dir, previous_result)

            # TODO: record summary of each node line to trial summary

    def __ingest(self, node_lines: Dict[str, List[Node]]):
        if any(list(map(lambda nodes: module_type_exists(nodes, 'bm25'), node_lines.values()))):
            # ingest BM25 corpus
            logger.info('Ingesting BM25 corpus...')
            bm25_dir = os.path.join(self.project_dir, 'resources', 'bm25.pkl')
            if not os.path.exists(os.path.dirname(bm25_dir)):
                os.makedirs(os.path.dirname(bm25_dir))
            if os.path.exists(bm25_dir):
                logger.info('BM25 corpus already exists.')
            else:
                bm25_ingest(bm25_dir, self.corpus_data)
            logger.info('BM25 corpus ingestion complete.')
            pass
        elif any(list(map(lambda nodes: module_type_exists(nodes, 'vector'), node_lines.values()))):
            # TODO: ingest vector DB
            pass
        else:
            logger.info('No ingestion needed.')

    def __get_new_trial_name(self) -> str:
        trial_json_path = os.path.join(self.project_dir, 'trial.json')
        if not os.path.exists(trial_json_path):
            return '0'
        with open(trial_json_path, 'r') as f:
            trial_json = json.load(f)
        return str(int(trial_json[-1]['trial_name']) + 1)

    def __make_trial_dir(self, trial_name: str):
        trial_json_path = os.path.join(self.project_dir, 'trial.json')
        if os.path.exists(trial_json_path):
            with open(trial_json_path, 'r') as f:
                trial_json = json.load(f)
        else:
            trial_json = []

        trial_json.append({
            'trial_name': trial_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        os.makedirs(os.path.join(self.project_dir, trial_name))
        with open(trial_json_path, 'w') as f:
            json.dump(trial_json, f, indent=4)

    @staticmethod
    def _load_node_lines(yaml_path: str) -> Dict[str, List[Node]]:
        if not os.path.exists(yaml_path):
            raise ValueError(f"YAML file {yaml_path} does not exist.")
        with open(yaml_path, 'r') as stream:
            try:
                yaml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(f"YAML file {yaml_path} could not be loaded.") from exc

        node_lines = yaml_dict['node_lines']
        node_line_dict = {}
        for node_line in node_lines:
            node_line_dict[node_line['node_line_name']] = list(
                map(lambda node: Node.from_dict(node), node_line['nodes']))
        return node_line_dict
