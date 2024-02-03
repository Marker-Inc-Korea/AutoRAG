
from typing import Callable, Dict
import importlib


def dynamically_find_function(key: str, target_dict: Dict) -> Callable:
    if key in target_dict:
        module_path, func_name = target_dict[key]
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return func
    else:
        raise KeyError(f"Key {key} is not supported.")


def get_support_modules(module_name: str) -> Callable:
    support_modules = {
        'query_decompose': ('autorag.nodes.queryexpansion', 'query_decompose'),
        'hyde': ('autorag.nodes.queryexpansion', 'hyde'),
        'bm25': ('autorag.nodes.retrieval', 'bm25'),
        'vectordb': ('autorag.nodes.retrieval', 'vectordb'),
        'fstring': ('autorag.nodes.promptmaker', 'fstring'),
        'llama_index_llm': ('autorag.nodes.generator', 'llama_index_llm'),
        'tree_summarize': ('autorag.nodes.passagecompressor', 'tree_summarize'),
        'monot5': ('autorag.nodes.passagereranker', 'monot5'),
        'tart': ('autorag.nodes.passagereranker', 'tart'),
        'upr': ('autorag.nodes.passagereranker', 'upr'),
        'hybrid_rrf': ('autorag.nodes.retrieval', 'hybrid_rrf'),
        'hybrid_cc': ('autorag.nodes.retrieval', 'hybrid_cc'),
    }
    return dynamically_find_function(module_name, support_modules)


def get_support_nodes(node_name: str) -> Callable:
    support_nodes = {
        'query_expansion': ('autorag.nodes.queryexpansion.run', 'run_query_expansion_node'),
        'retrieval': ('autorag.nodes.retrieval.run', 'run_retrieval_node'),
        'generator': ('autorag.nodes.generator.run', 'run_generator_node'),
        'prompt_maker': ('autorag.nodes.promptmaker.run', 'run_prompt_maker_node'),
        'passage_compressor': ('autorag.nodes.passagecompressor.run', 'run_passage_compressor_node'),
        'passage_reranker': ('autorag.nodes.passagereranker.run', 'run_passage_reranker_node'),
    }
    return dynamically_find_function(node_name, support_nodes)
