import pandas as pd

from autorag.deploy import summary_df_to_yaml


def test_summary_df_to_yaml():
    summary_df = pd.DataFrame({
        'node_line_name': ['node_line_1', 'node_line_1', 'node_line_2'],
        'node_type': ['retrieval', 'rerank', 'generation'],
        'best_module_filename': ['bm25=>top_k_50.parquet', 'upr=>model_llama-2-havertz_chelsea.parquet',
                                 'gpt-4=>top_p_0.9.parquet'],
        'best_module_name': ['bm25', 'upr', 'gpt-4'],
        'best_module_params': [{'top_k': 50}, {'model': 'llama-2', 'havertz': 'chelsea'}, {'top_p': 0.9}],
        'best_execution_time': [1.0, 0.5, 2.0]
    })
    yaml_dict = summary_df_to_yaml(summary_df)
    assert yaml_dict == {
        'node_lines': [
            {
                'node_line_name': 'node_line_1',
                'nodes': [
                    {
                        'node_type': 'retrieval',
                        'modules': [
                            {
                                'module_type': 'bm25',
                                'top_k': 50
                            }
                        ]
                    },
                    {
                        'node_type': 'rerank',
                        'modules': [
                            {
                                'module_type': 'upr',
                                'model': 'llama-2',
                                'havertz': 'chelsea'
                            }
                        ]
                    }
                ]
            },
            {
                'node_line_name': 'node_line_2',
                'nodes': [
                    {
                        'node_type': 'generation',
                        'modules': [
                            {
                                'module_type': 'gpt-4',
                                'top_p': 0.9
                            }
                        ]
                    }
                ]
            }
        ]
    }
