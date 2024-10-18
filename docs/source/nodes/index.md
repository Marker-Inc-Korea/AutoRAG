---
myst:
  html_meta:
    title: AutoRAG - Available List
    description: All nodes and modules of AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,modules,nodes
---
# Available List

📌 You can check our all supporting Nodes & modules
<table class="table table-my-special-table">
  <tr>
    <th>Nodes</th>
    <th>Modules</th>
    <th>YAML module names</th>
  </tr>
  <tr>
    <td rowspan="4"><a href="https://marker-inc-korea.github.io/AutoRAG/nodes/query_expansion/query_expansion.html">1️⃣ Query Expansion</a></td>
    <td>Query Decompose</td>
    <td>query_decompose</td>
  </tr>
  <tr>
    <td>HyDE</td>
    <td>hyde</td>
  </tr>
  <tr>
    <td>Multi Query Expansion</td>
    <td>multi_query_expansion</td>
  </tr>
  <tr>
    <td>📌 Pass Module pass_query_expansion</td>
    <td>pass_query_expansion</td>
  </tr>
  <tr>
    <td rowspan="4">2️⃣ Retrieval</td>
    <td>BM25 VectorDB (choose embedding model)</td>
    <td>bm25</td>
  </tr>
  <tr>
    <td>📌 Hybrid Module Hybrid with rrf (reciprocal rank fusion)</td>
    <td>hybrid_rrf</td>
  </tr>
  <tr>
    <td>Hybrid with cc (convex combination) w. four different normalize methods</td>
    <td>hybrid_cc</td>
  </tr>
  <tr>
    <td>Vectordb</td>
    <td>vectordb</td>
  </tr>
  <tr>
    <td rowspan="2">3️⃣ Passage Augmenter</td>
    <td>Prev Next Augmenter</td>
    <td>prev_next_augmenter</td>
  </tr>
  <tr>


table.table.table-my-special-table {
    border-collapse: collapse; 
    border:1px solid #69899F;
} 
table.table.table-my-special-table td{
    border:1px dotted #000000;
    padding:5px;
}
table.table.table-my-special-table td:first-child{
    border-left:0px solid #000000;
}
table.table.table-my-special-table th{
   border:2px solid #69899F;
   padding:5px;
}


  </tr>
  <tr>
    <td rowspan="16">4️⃣ Passage Reranker</td>
    <td>UPR</td>
    <td>upr</td>
  </tr>
  <tr>
    <td>Tart</td>
    <td>tart</td>
  </tr>
  <tr>
    <td>MonoT5</td>
    <td>monot5</td>
  </tr>
  <tr>
    <td>Cohere reranker</td>
    <td>cohere_reranker</td>
  </tr>
  <tr>
    <td>RankGPT</td>
    <td>rankgpt</td>
  </tr>
  <tr>
    <td>Jina Reranker</td>
    <td>jina_reranker</td>
  </tr>
  <tr>
    <td>Sentence Transformer Reranker</td>
    <td>sentence_transformer_reranker</td>
  </tr>
  <tr>
    <td>Colbert Reranker</td>
    <td>colbert_reranker</td>
  </tr>
  <tr>
    <td>Flag Embedding Reranker</td>
    <td>flag_embedding_reranker</td>
  </tr>
  <tr>
    <td>Flag Embedding LLM Reranker</td>
    <td>flag_embedding_llm_reranker</td>
  </tr>
  <tr>
    <td>Time Reranker</td>
    <td>time_reranker</td>
  </tr>
  <tr>
    <td>Open VINO Reranker</td>
    <td>open_vino_reranker</td>
  </tr>
  <tr>
    <td>VoyageAI Reranker</td>
    <td>voyageai_reranker</td>
  </tr>
  <tr>
    <td>MixedBread AI Reranker</td>
    <td>mixedbread_ai_reranker</td>
  </tr>
  <tr>
    <td>📌 Only for Korean Ko-reranker</td>
    <td>ko_reranker</td>
  </tr>
  <tr>
    <td>📌 Pass Module pass_reranker</td>
    <td>pass_reranker</td>
  </tr>
  <tr>
    <td rowspan="6">5️⃣ Passage Filter</td>
    <td>Similarity threshold cutoff</td>
    <td>similarity_threshold_cutoff</td>
  </tr>
  <tr>
    <td>Similarity percentile cutoff</td>
    <td>similarity_percentile_cutoff</td>
  </tr>
  <tr>
    <td>Recency filter</td>
    <td>recency_filter</td>
  </tr>
  <tr>
    <td>Threshold cutoff</td>
    <td>threshold_cutoff</td>
  </tr>
  <tr>
    <td>Percentile cutoff</td>
    <td>percentile_cutoff</td>
  </tr>
  <tr>
    <td>📌 Pass Module pass_passage_filter</td>
    <td>pass_passage_filter</td>
  </tr>
  <tr>
    <td rowspan="3">6️⃣ Passage Compressor</td>
    <td>Tree Summarize</td>
    <td>tree_summarize</td>
  </tr>
  <tr>
    <td>Refine</td>
    <td>refine</td>
  </tr>
  <tr>
    <td>📌 Pass Module pass_compressor</td>
    <td>pass_compressor</td>
  </tr>
  <tr>
    <td rowspan="3">7️⃣ Prompt Maker</td>
    <td>Default Prompt Maker (f-string)</td>
    <td>fstring</td>
  </tr>
  <tr>
    <td>Long Context Reorder</td>
    <td>long_context_reorder</td>
  </tr>
  <tr>
    <td>Window Replacement</td>
    <td>window_replacement</td>
  </tr>
  <tr>
    <td rowspan="3">8️⃣ Generator</td>
    <td>llama index llm</td>
    <td>llama_index_llm</td>
  </tr>
  <tr>
    <td>openai_llm</td>
    <td>openai_llm</td>
  </tr>
  <tr>
    <td>vllm</td>
    <td>vllm</td>
  </tr>
</table>

For a comprehensive and up-to-date list of supporting nodes and modules, please refer to our dedicated Notion page. You can access this information [here](https://edai.notion.site/Supporting-Nodes-modules-0ebc7810649f4e41aead472a92976be4?pvs=4).