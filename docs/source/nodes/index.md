---
myst:
  html_meta:
    title: AutoRAG - Available List
    description: All nodes and modules of AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,modules,nodes
---
# Available List

📌 You can check our all supporting Nodes & modules


<style>
table.table.table-my-special-table {
    border-collapse: collapse; 
    border:1px solid #69899F;
} 
table.table.table-my-special-table td{
    border:1px dotted #69899F;
    padding:5px;
}
table.table.table-my-special-table td:first-child{
    border-left:0px solid #69899F;
}
table.table.table-my-special-table th{
   border:2px solid #69899F;
   padding:5px;
}

</style>

<table class="table table-my-special-table">
  <tbody>
    <tr>
      <td>Nodes</td>
      <td>Modules</td>
      <td>YAML module names</td>
    </tr>
    <tr>
      <td>1️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/query_expansion/query_expansion.html">Query Expansion</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/query_expansion/query_decompose.html">Query Decompose</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/query_expansion/hyde.html">HyDE</a>, <a href="https://docs.auto-rag.com/nodes/query_expansion/multi_query_expansion.html">Multi Query Expansion</a>, 📌 Pass Module: pass_query_expansion</td>
      <td>query_decompose, hyde, multi_query_expansion, pass_query_expansion</td>
    </tr>
    <tr>
      <td>2️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/retrieval/retrieval.html">Retrieval</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/retrieval/bm25.html">BM25</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/retrieval/vectordb.html">VectorDB</a>, 📌 Hybrid Module: hybrid_rrf, hybrid_cc</td>
      <td>bm25, vectordb, hybrid_rrf, hybrid_cc</td>
    </tr>
    <tr>
      <td>3️⃣ Passage Augmenter</td>
      <td>Prev Next Augmenter, 📌 Pass Module: pass_passage_augmenter</td>
      <td>prev_next_augmenter, pass_passage_augmenter</td>
    </tr>
    <tr>
      <td>4️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/passage_reranker.html">Passage Reranker</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/upr.html">UPR</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/tart.html">Tart</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/monot5.html">MonoT5</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/cohere.html">Cohere reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/rankgpt.html">RankGPT</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/jina_reranker.html">Jina Reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/sentence_transformer_reranker.html">Sentence Transformer Reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/colbert.html">Colbert Reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/flag_embedding_reranker.html">Flag Embedding Reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/flag_embedding_llm_reranker.html">Flag Embedding LLM Reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/time_reranker.html">Time Reranker</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_reranker/koreranker.html">Ko-reranker</a>, 📌 Pass Module: pass_reranker</td>
      <td>upr, tart, monot5, cohere_reranker, rankgpt, jina_reranker, sentence_transformer_reranker, colbert_reranker, flag_embedding_reranker, flag_embedding_llm_reranker, time_reranker, ko_reranker, pass_reranker</td>
    </tr>
    <tr>
      <td>5️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_filter/passage_filter.html">Passage Filter</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_filter/similarity_threshold_cutoff.html">similarity threshold cutoff</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_filter/similarity_percentile_cutoff.html">similarity percentile cutoff</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_filter/recency_filter.html">recency filter</a>, 📌 Pass Module: pass_passage_filter</td>
      <td>similarity_threshold_cutoff, similarity_percentile_cutoff, recency_filter, pass_passage_filter</td>
    </tr>
    <tr>
      <td>6️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_compressor/passage_compressor.html">Passage Compressor</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_compressor/tree_summarize.html">Tree Summarize</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/passage_compressor/refine.html">Refine</a>, 📌 Pass Module: pass_compressor</td>
      <td>tree_summarize, refine, pass_compressor</td>
    </tr>
    <tr>
      <td>7️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/prompt_maker/prompt_maker.html">Prompt Maker</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/prompt_maker/fstring.html">Default Prompt Maker (f-string)</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/prompt_maker/long_context_reorder.html">Long Context Reorder</a>, <a href="https://docs.auto-rag.com/nodes/prompt_maker/window_replacement.html">Window Replacement</a></td>
      <td>fstring, long_context_reorder, window_replacement</td>
    </tr>
    <tr>
      <td>8️⃣ <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/generator/generator.html">Generator</a></td>
      <td><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/generator/llama_index_llm.html">llama index llm</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/generator/openai_llm.html">openai_llm</a>, <a href="https://marker-inc-korea.github.io/AutoRAG/nodes/generator/vllm.html">vllm</a></td>
      <td>llama_index_llm, openai_llm, vllm</td>
    </tr>
  </tbody>
</table>

For a comprehensive and up-to-date list of supporting nodes and modules, please refer to our dedicated Notion page. You can access this information [here](https://edai.notion.site/Supporting-Nodes-modules-0ebc7810649f4e41aead472a92976be4?pvs=4).