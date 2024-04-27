from typing import List

import pandas as pd

from autorag.nodes.passageaugmenter.base import passage_augmenter_node


@passage_augmenter_node
def prev_next_augmenter(ids_list: List[List[str]],
                        corpus_df: pd.DataFrame,
                        num_passages: int = 1,
                        mode: str = 'both'
                        ) -> List[List[str]]:
    """
    Add passages before and/or after the retrieved passage.
    For more information, visit https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/PrevNextPostprocessorDemo/.

    :param ids_list: The list of lists of ids retrieved
    :param corpus_df: The corpus dataframe
    :param num_passages: The number of passages to add before and after the retrieved passage
        Default is 1.
    :param mode: The mode of augmentation
        'prev': add passages before the retrieved passage
        'next': add passages after the retrieved passage
        'both': add passages before and after the retrieved passage
        Default is 'next'.
    :return: The list of lists of augmented ids
    """
    if mode not in ['prev', 'next', 'both']:
        raise ValueError(f"mode must be 'prev', 'next', or 'both', but got {mode}")

    augmented_ids = [(lambda ids: prev_next_augmenter_pure(ids, corpus_df, mode, num_passages))(ids) for ids in
                     ids_list]

    return augmented_ids


def prev_next_augmenter_pure(ids: List[str], corpus_df: pd.DataFrame, mode: str, num_passages: int):
    def fetch_id_sequence(start_id, key):
        sequence = []
        current_id = start_id
        for _ in range(num_passages):
            current_id = corpus_df.loc[corpus_df['doc_id'] == current_id]['metadata'].values[0].get(key)
            if current_id is None:
                break
            sequence.append(current_id)
        return sequence

    augmented_group = []
    for id_ in ids:
        current_ids = [id_]
        if mode in ['prev', 'both']:
            current_ids = fetch_id_sequence(id_, 'prev_id')[::-1] + current_ids
        if mode in ['next', 'both']:
            current_ids += fetch_id_sequence(id_, 'next_id')
        augmented_group.extend(current_ids)
    return augmented_group
