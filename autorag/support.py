import importlib
from typing import Callable, Dict


def dynamically_find_function(key: str, target_dict: Dict) -> Callable:
    if key in target_dict:
        module_path, func_name = target_dict[key]
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return func
    else:
        raise KeyError(f"Input module or node {key} is not supported.")


def get_support_modules(module_name: str) -> Callable:
    support_modules = {
        # query_expansion
        'query_decompose': ('autorag.nodes.queryexpansion', 'query_decompose'),
        'hyde': ('autorag.nodes.queryexpansion', 'hyde'),
        'pass_query_expansion': ('autorag.nodes.queryexpansion', 'pass_query_expansion'),
        # retrieval
        'bm25': ('autorag.nodes.retrieval', 'bm25'),
        'vectordb': ('autorag.nodes.retrieval', 'vectordb'),
        'hybrid_rrf': ('autorag.nodes.retrieval', 'hybrid_rrf'),
        'hybrid_cc': ('autorag.nodes.retrieval', 'hybrid_cc'),
        'hybrid_rsf': ('autorag.nodes.retrieval', 'hybrid_rsf'),
        'hybrid_dbsf': ('autorag.nodes.retrieval', 'hybrid_dbsf'),
        # passage_augmenter
        'prev_next_augmenter': ('autorag.nodes.passageaugmenter', 'prev_next_augmenter'),
        # passage_reranker
        'monot5': ('autorag.nodes.passagereranker', 'monot5'),
        'tart': ('autorag.nodes.passagereranker', 'tart'),
        'upr': ('autorag.nodes.passagereranker', 'upr'),
        'koreranker': ('autorag.nodes.passagereranker', 'koreranker'),
        'pass_reranker': ('autorag.nodes.passagereranker', 'pass_reranker'),
        'cohere_reranker': ('autorag.nodes.passagereranker', 'cohere_reranker'),
        'rankgpt': ('autorag.nodes.passagereranker', 'rankgpt'),
        'jina_reranker': ('autorag.nodes.passagereranker', 'jina_reranker'),
        'colbert_reranker': ('autorag.nodes.passagereranker', 'colbert_reranker'),
        'sentence_transformer_reranker': ('autorag.nodes.passagereranker', 'sentence_transformer_reranker'),
        'flag_embedding_reranker': ('autorag.nodes.passagereranker', 'flag_embedding_reranker'),
        'flag_embedding_llm_reranker': ('autorag.nodes.passagereranker', 'flag_embedding_llm_reranker'),
        'time_reranker': ('autorag.nodes.passagereranker', 'time_reranker'),
        # passage_filter
        'pass_passage_filter': ('autorag.nodes.passagefilter', 'pass_passage_filter'),
        'similarity_threshold_cutoff': ('autorag.nodes.passagefilter', 'similarity_threshold_cutoff'),
        'similarity_percentile_cutoff': ('autorag.nodes.passagefilter', 'similarity_percentile_cutoff'),
        'recency_filter': ('autorag.nodes.passagefilter', 'recency_filter'),
        # passage_compressor
        'tree_summarize': ('autorag.nodes.passagecompressor', 'tree_summarize'),
        'pass_compressor': ('autorag.nodes.passagecompressor', 'pass_compressor'),
        'refine': ('autorag.nodes.passagecompressor', 'refine'),
        # prompt_maker
        'fstring': ('autorag.nodes.promptmaker', 'fstring'),
        'long_context_reorder': ('autorag.nodes.promptmaker', 'long_context_reorder'),
        # generator
        'llama_index_llm': ('autorag.nodes.generator', 'llama_index_llm'),
        'vllm': ('autorag.nodes.generator', 'vllm'),
    }
    return dynamically_find_function(module_name, support_modules)


def get_support_nodes(node_name: str) -> Callable:
    support_nodes = {
        'query_expansion': ('autorag.nodes.queryexpansion.run', 'run_query_expansion_node'),
        'retrieval': ('autorag.nodes.retrieval.run', 'run_retrieval_node'),
        'generator': ('autorag.nodes.generator.run', 'run_generator_node'),
        'prompt_maker': ('autorag.nodes.promptmaker.run', 'run_prompt_maker_node'),
        'passage_filter': ('autorag.nodes.passagefilter.run', 'run_passage_filter_node'),
        'passage_compressor': ('autorag.nodes.passagecompressor.run', 'run_passage_compressor_node'),
        'passage_reranker': ('autorag.nodes.passagereranker.run', 'run_passage_reranker_node'),
        'passage_augmenter': ('autorag.nodes.passageaugmenter.run', 'run_passage_augmenter_node'),
    }
    return dynamically_find_function(node_name, support_nodes)
