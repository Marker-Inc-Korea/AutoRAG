from typing import List, Tuple

from llama_index.core.node_parser import SentenceSplitter

from autorag.nodes.retrieval.base import retrieval_node
from autorag.nodes.retrieval.bm25 import bm25, bm25_ingest


@retrieval_node
def recursive_chunk(queries: List[List[str]],
                    ids: List[List[str]],
                    contents_list: List[List[str]],
                    baseline_retrieval: str,
                    top_k: int,
                    sub_chunk_sizes: List[int] = [128, 256, 512],
                    chunk_overlap: int = 20,
                    ) -> Tuple[List[List[str]], List[List[float]]]:
    sub_sentence_splitters = [SentenceSplitter(chunk_size=c, chunk_overlap=chunk_overlap) for c in sub_chunk_sizes]
    for sub_sentence_splitter in sub_sentence_splitters:
        for contents in contents_list:
            for content in contents:
                split_result = sub_sentence_splitter.split_text(content)
            # Metric은 어떻게 측정하지 ..? 새롭게 embedding이 들어가면 id도 새로..?
            # 1. qa retrieval_gt에 id를 추가한다.

    if baseline_retrieval == "bm25":
        bm25_ingest()
        original_bm25 = bm25.__wrapped__
        id_result, score_result = original_bm25(queries, top_k=top_k, bm25_corpus=bm25_corpus)
    elif baseline_retrieval == "vectordb":
        pass
    else:
        raise ValueError(f"Invalid baseline_retrieval: {baseline_retrieval}")
