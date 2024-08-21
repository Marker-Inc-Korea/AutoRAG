import os
import pathlib

import pandas as pd
import pytest

from autorag import dashboard
from autorag.dashboard import get_metric_values, make_trial_summary_md

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
sample_project_dir = os.path.join(root_dir, "resources", "result_project")
sample_trial_dir = os.path.join(sample_project_dir, "0")


@pytest.fixture
def retrieval_summary_df():
	return pd.read_csv(
		os.path.join(sample_trial_dir, "retrieve_node_line", "retrieval", "summary.csv")
	)


def test_get_metric_values(retrieval_summary_df):
	result_dict = get_metric_values(retrieval_summary_df)
	assert len(result_dict.keys()) == 3
	assert set(list(result_dict.keys())) == {
		"retrieval_f1",
		"retrieval_recall",
		"retrieval_precision",
	}
	assert result_dict["retrieval_recall"] == 1.0
	assert result_dict["retrieval_precision"] == 0.1


def test_make_trial_summary_md():
	md_text = make_trial_summary_md(sample_trial_dir)
	assert bool(md_text)


@pytest.mark.skip(
	reason="Can't stop this test on the github action or pytest cli setup"
)
def test_dashboard_run():
	dashboard.run(sample_trial_dir)
