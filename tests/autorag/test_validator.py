import os
import pathlib

from autorag.validator import Validator

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")


def test_validator_validate():
	qa_path = os.path.join(resource_dir, "qa_data_sample.parquet")
	corpus_path = os.path.join(resource_dir, "corpus_data_sample.parquet")
	yaml_path = os.path.join(resource_dir, "simple_mock.yaml")

	validator = Validator(qa_path, corpus_path)
	validator.validate(yaml_path, qa_cnt=3, random_state=42)
