import os
import pathlib
import subprocess
import tempfile

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, 'resources')


def test_evaluator_cli():
    os.environ['BM25'] = 'bm25'
    with tempfile.TemporaryDirectory() as project_dir:
        result = subprocess.run(['autorag', 'evaluate', '--config', os.path.join(resource_dir, 'simple.yaml'),
                                 '--qa_data_path', os.path.join(resource_dir, 'qa_data_sample.parquet'),
                                 '--corpus_data_path', os.path.join(resource_dir, 'corpus_data_sample.parquet'),
                                 '--project_dir', project_dir])
        assert result.returncode == 0
        # check if the files are created
        assert os.path.exists(os.path.join(project_dir, '0'))
        assert os.path.exists(os.path.join(project_dir, 'data'))
        assert os.path.exists(os.path.join(project_dir, 'resources'))
        assert os.path.exists(os.path.join(project_dir, 'trial.json'))
        assert os.path.exists(os.path.join(project_dir, '0', 'retrieve_node_line'))
        assert os.path.exists(os.path.join(project_dir, '0', 'retrieve_node_line', 'retrieval'))
        assert os.path.exists(os.path.join(project_dir, '0', 'retrieve_node_line', 'retrieval', '0.parquet'))
