from typing import List

from autorag.nodes.passageaugmenter.base import passage_augmenter_node


@passage_augmenter_node
def prev_next_augmenter(ids_list: List[List[str]],
                        all_ids_list: List[str],
                        num_passages: int = 1,
                        mode: str = 'next'
                        ) -> List[List[str]]:
    """
    Add passages before and/or after the retrieved passage.
    For more information, visit https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/PrevNextPostprocessorDemo/.

    :param ids_list: The list of lists of ids retrieved
    :param all_ids_list: The list of all ids in the corpus
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
        raise ValueError(f"mode must be 'prev' or 'next', but got {mode}")

    augmented_ids = []

    id_index_map = {id_: idx for idx, id_ in enumerate(all_ids_list)}

    for sublist in ids_list:
        new_sublist = []
        for id_ in sublist:
            if id_ in id_index_map:
                idx = id_index_map[id_]

                if mode == 'prev' or mode == 'both':
                    start_idx = max(0, idx - num_passages)
                    new_sublist.extend(all_ids_list[start_idx:idx])

                new_sublist.append(id_)

                if mode == 'next' or mode == 'both':
                    end_idx = min(len(all_ids_list), idx + 1 + num_passages)
                    new_sublist.extend(all_ids_list[idx + 1:end_idx])

        augmented_ids.append(new_sublist)

    return augmented_ids
