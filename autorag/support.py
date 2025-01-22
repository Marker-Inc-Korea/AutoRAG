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
		# parse
		"langchain_parse": ("autorag.data.parse", "langchain_parse"),
		"clova": ("autorag.data.parse.clova", "clova_ocr"),
		"llamaparse": ("autorag.data.parse.llamaparse", "llama_parse"),
		"table_hybrid_parse": (
			"autorag.data.parse.table_hybrid_parse",
			"table_hybrid_parse",
		),
		# chunk
		"llama_index_chunk": ("autorag.data.chunk", "llama_index_chunk"),
		"langchain_chunk": ("autorag.data.chunk", "langchain_chunk"),
		# query_expansion
		"query_decompose": ("autorag.nodes.queryexpansion", "QueryDecompose"),
		"hyde": ("autorag.nodes.queryexpansion", "HyDE"),
		"pass_query_expansion": (
			"autorag.nodes.queryexpansion",
			"PassQueryExpansion",
		),
		"multi_query_expansion": (
			"autorag.nodes.queryexpansion",
			"MultiQueryExpansion",
		),
		"QueryDecompose": ("autorag.nodes.queryexpansion", "QueryDecompose"),
		"HyDE": ("autorag.nodes.queryexpansion", "HyDE"),
		"PassQueryExpansion": (
			"autorag.nodes.queryexpansion",
			"PassQueryExpansion",
		),
		"MultiQueryExpansion": (
			"autorag.nodes.queryexpansion",
			"MultiQueryExpansion",
		),
		# retrieval
		"bm25": ("autorag.nodes.retrieval", "BM25"),
		"BM25": ("autorag.nodes.retrieval", "BM25"),
		"vectordb": ("autorag.nodes.retrieval", "VectorDB"),
		"VectorDB": ("autorag.nodes.retrieval", "VectorDB"),
		"hybrid_rrf": ("autorag.nodes.retrieval", "HybridRRF"),
		"HybridRRF": ("autorag.nodes.retrieval", "HybridRRF"),
		"hybrid_cc": ("autorag.nodes.retrieval", "HybridCC"),
		"HybridCC": ("autorag.nodes.retrieval", "HybridCC"),
		# passage_augmenter
		"prev_next_augmenter": (
			"autorag.nodes.passageaugmenter",
			"PrevNextPassageAugmenter",
		),
		"PrevNextPassageAugmenter": (
			"autorag.nodes.passageaugmenter",
			"PrevNextPassageAugmenter",
		),
		"pass_passage_augmenter": (
			"autorag.nodes.passageaugmenter",
			"PassPassageAugmenter",
		),
		"PassPassageAugmenter": (
			"autorag.nodes.passageaugmenter",
			"PassPassageAugmenter",
		),
		# passage_reranker
		"monot5": ("autorag.nodes.passagereranker", "MonoT5"),
		"MonoT5": ("autorag.nodes.passagereranker", "MonoT5"),
		"tart": ("autorag.nodes.passagereranker.tart", "Tart"),
		"Tart": ("autorag.nodes.passagereranker.tart", "Tart"),
		"upr": ("autorag.nodes.passagereranker", "Upr"),
		"Upr": ("autorag.nodes.passagereranker", "Upr"),
		"koreranker": ("autorag.nodes.passagereranker", "KoReranker"),
		"KoReranker": ("autorag.nodes.passagereranker", "KoReranker"),
		"pass_reranker": ("autorag.nodes.passagereranker", "PassReranker"),
		"PassReranker": ("autorag.nodes.passagereranker", "PassReranker"),
		"cohere_reranker": ("autorag.nodes.passagereranker", "CohereReranker"),
		"CohereReranker": ("autorag.nodes.passagereranker", "CohereReranker"),
		"rankgpt": ("autorag.nodes.passagereranker", "RankGPT"),
		"RankGPT": ("autorag.nodes.passagereranker", "RankGPT"),
		"jina_reranker": ("autorag.nodes.passagereranker", "JinaReranker"),
		"JinaReranker": ("autorag.nodes.passagereranker", "JinaReranker"),
		"colbert_reranker": ("autorag.nodes.passagereranker", "ColbertReranker"),
		"ColbertReranker": ("autorag.nodes.passagereranker", "ColbertReranker"),
		"sentence_transformer_reranker": (
			"autorag.nodes.passagereranker",
			"SentenceTransformerReranker",
		),
		"SentenceTransformerReranker": (
			"autorag.nodes.passagereranker",
			"SentenceTransformerReranker",
		),
		"flag_embedding_reranker": (
			"autorag.nodes.passagereranker",
			"FlagEmbeddingReranker",
		),
		"FlagEmbeddingReranker": (
			"autorag.nodes.passagereranker",
			"FlagEmbeddingReranker",
		),
		"flag_embedding_llm_reranker": (
			"autorag.nodes.passagereranker",
			"FlagEmbeddingLLMReranker",
		),
		"FlagEmbeddingLLMReranker": (
			"autorag.nodes.passagereranker",
			"FlagEmbeddingLLMReranker",
		),
		"time_reranker": ("autorag.nodes.passagereranker", "TimeReranker"),
		"TimeReranker": ("autorag.nodes.passagereranker", "TimeReranker"),
		"openvino_reranker": ("autorag.nodes.passagereranker", "OpenVINOReranker"),
		"OpenVINOReranker": ("autorag.nodes.passagereranker", "OpenVINOReranker"),
		"voyageai_reranker": ("autorag.nodes.passagereranker", "VoyageAIReranker"),
		"VoyageAIReranker": ("autorag.nodes.passagereranker", "VoyageAIReranker"),
		"mixedbreadai_reranker": (
			"autorag.nodes.passagereranker",
			"MixedbreadAIReranker",
		),
		"MixedbreadAIReranker": (
			"autorag.nodes.passagereranker",
			"MixedbreadAIReranker",
		),
		"flashrank_reranker": ("autorag.nodes.passagereranker", "FlashRankReranker"),
		"FlashRankReranker": ("autorag.nodes.passagereranker", "FlashRankReranker"),
		# passage_filter
		"pass_passage_filter": ("autorag.nodes.passagefilter", "PassPassageFilter"),
		"similarity_threshold_cutoff": (
			"autorag.nodes.passagefilter",
			"SimilarityThresholdCutoff",
		),
		"similarity_percentile_cutoff": (
			"autorag.nodes.passagefilter",
			"SimilarityPercentileCutoff",
		),
		"recency_filter": ("autorag.nodes.passagefilter", "RecencyFilter"),
		"threshold_cutoff": ("autorag.nodes.passagefilter", "ThresholdCutoff"),
		"percentile_cutoff": ("autorag.nodes.passagefilter", "PercentileCutoff"),
		"PassPassageFilter": ("autorag.nodes.passagefilter", "PassPassageFilter"),
		"SimilarityThresholdCutoff": (
			"autorag.nodes.passagefilter",
			"SimilarityThresholdCutoff",
		),
		"SimilarityPercentileCutoff": (
			"autorag.nodes.passagefilter",
			"SimilarityPercentileCutoff",
		),
		"RecencyFilter": ("autorag.nodes.passagefilter", "RecencyFilter"),
		"ThresholdCutoff": ("autorag.nodes.passagefilter", "ThresholdCutoff"),
		"PercentileCutoff": ("autorag.nodes.passagefilter", "PercentileCutoff"),
		# passage_compressor
		"tree_summarize": ("autorag.nodes.passagecompressor", "TreeSummarize"),
		"pass_compressor": ("autorag.nodes.passagecompressor", "PassCompressor"),
		"refine": ("autorag.nodes.passagecompressor", "Refine"),
		"longllmlingua": ("autorag.nodes.passagecompressor", "LongLLMLingua"),
		"TreeSummarize": ("autorag.nodes.passagecompressor", "TreeSummarize"),
		"Refine": ("autorag.nodes.passagecompressor", "Refine"),
		"LongLLMLingua": ("autorag.nodes.passagecompressor", "LongLLMLingua"),
		"PassCompressor": ("autorag.nodes.passagecompressor", "PassCompressor"),
		# prompt_maker
		"fstring": ("autorag.nodes.promptmaker", "Fstring"),
		"long_context_reorder": ("autorag.nodes.promptmaker", "LongContextReorder"),
		"window_replacement": ("autorag.nodes.promptmaker", "WindowReplacement"),
		"Fstring": ("autorag.nodes.promptmaker", "Fstring"),
		"LongContextReorder": ("autorag.nodes.promptmaker", "LongContextReorder"),
		"WindowReplacement": ("autorag.nodes.promptmaker", "WindowReplacement"),
		# generator
		"llama_index_llm": ("autorag.nodes.generator", "LlamaIndexLLM"),
		"vllm": ("autorag.nodes.generator", "Vllm"),
		"openai_llm": ("autorag.nodes.generator", "OpenAILLM"),
		"vllm_api": ("autorag.nodes.generator", "VllmAPI"),
		"LlamaIndexLLM": ("autorag.nodes.generator", "LlamaIndexLLM"),
		"Vllm": ("autorag.nodes.generator", "Vllm"),
		"OpenAILLM": ("autorag.nodes.generator", "OpenAILLM"),
		"VllmAPI": ("autorag.nodes.generator", "VllmAPI"),
	}
	return dynamically_find_function(module_name, support_modules)


def get_support_nodes(node_name: str) -> Callable:
	support_nodes = {
		"query_expansion": (
			"autorag.nodes.queryexpansion.run",
			"run_query_expansion_node",
		),
		"retrieval": ("autorag.nodes.retrieval.run", "run_retrieval_node"),
		"generator": ("autorag.nodes.generator.run", "run_generator_node"),
		"prompt_maker": ("autorag.nodes.promptmaker.run", "run_prompt_maker_node"),
		"passage_filter": (
			"autorag.nodes.passagefilter.run",
			"run_passage_filter_node",
		),
		"passage_compressor": (
			"autorag.nodes.passagecompressor.run",
			"run_passage_compressor_node",
		),
		"passage_reranker": (
			"autorag.nodes.passagereranker.run",
			"run_passage_reranker_node",
		),
		"passage_augmenter": (
			"autorag.nodes.passageaugmenter.run",
			"run_passage_augmenter_node",
		),
	}
	return dynamically_find_function(node_name, support_nodes)
