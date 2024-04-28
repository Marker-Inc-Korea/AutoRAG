from typing import List, Tuple

from llama_index.core.node_parser import SentenceSplitter

from autorag.nodes.passageaugmenter.base import passage_augmenter_node


@passage_augmenter_node
def recursive_chunk(ids_list: List[List[str]],
                    contents_list: List[List[str]],
                    sub_chunk_sizes: List[int] = [128, 256, 512],
                    chunk_overlap: int = 20,
                    ) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Add passages by recursively chunking the retrieved passage.
    For more information, visit https://docs.llamaindex.ai/en/stable/examples/retrievers/recurisve_retriever_nodes_braintrust/#chunk-references-smaller-child-chunks-referring-to-bigger-parent-chunk

    :param ids_list: The list of lists of ids retrieved
    :param contents_list: The list of lists of contents retrieved
    :param sub_chunk_sizes: The list of chunk sizes to split the retrieved passage
        Default is [128, 256, 512].
    :param chunk_overlap: The overlap between the chunks
        Default is 20.
    :return: Tuple of lists containing the augmented ids and contents
    """
    sub_sentence_splitters = [SentenceSplitter(chunk_size=c, chunk_overlap=chunk_overlap) for c in sub_chunk_sizes]
    augmented_ids, augmented_contents = [], []

    for ids, contents in zip(ids_list, contents_list):
        sublist_ids, split_contents = recursive_chunk_pure(ids, contents, sub_sentence_splitters)
        augmented_contents.append(split_contents)
        augmented_ids.append(sublist_ids)

    return augmented_ids, augmented_contents


def recursive_chunk_pure(ids: List[str], contents: List[str],
                         sub_sentence_splitters: List[SentenceSplitter]
                         ) -> Tuple[List[str], List[str]]:
    sublist_ids, split_contents = [], []
    for id_, content in zip(ids, contents):
        for sub_sentence_splitter in sub_sentence_splitters:
            split_results = sub_sentence_splitter.split_text(content)
            split_contents.extend(split_results)
            sublist_ids.extend([id_] * len(split_results))
    return sublist_ids, split_contents
