# Model Card: qwen3-embedding-0.6b

This card covers the `qwen3-embedding-0.6b` embedding profile, the default embedder AutoRAG caches for its loopback-only local gateway. AutoRAG does not train models, does not host a Hugging Face Space, and does not publish user corpora. It fetches one pinned GGUF file and serves embeddings on 127.0.0.1.

## Intended use

Local, loopback-only embedding for AutoRAG's MinSync and native-datasource semantic retrieval. The librarian agent embeds document chunks and queries on the operator's own machine. Corpus text must not be sent to a remote embedding endpoint, and AutoRAG's default configuration never does.

## Unsuitable use

- Chat or text generation of any kind. This model produces vectors, not tokens.
- Treating embedding similarity as ground-truth classification or fact verification.
- Biometric identification, surveillance, or profiling of people.
- Training or fine-tuning a new model from user corpora inside AutoRAG.
- Shipping the weights inside the npm package. Release jobs fetch the pinned URL and verify its SHA-256 instead.
- Substituting a different GGUF build than the pinned artifact below.

## Limitations

- This is an embedding model, not a generative or chat model. It cannot answer questions, summarize, or produce text.
- Output dimension is 1024 as configured in AutoRAG. The upstream model supports smaller truncated dimensions, but AutoRAG pins 1024.
- The AutoRAG profile uses an empty query prefix and an empty passage prefix. The upstream card notes that instruction-style query prompts can improve retrieval scores slightly, but AutoRAG does not apply them in this profile.
- Per the upstream card, the Qwen3 Embedding series supports over 100 languages, including programming languages. Language-by-language quality varies; the upstream card does not break coverage down per language.
- 0.6B parameters is the smallest size in the series. It trades peak benchmark accuracy for speed and footprint, which fits AutoRAG's local-first design.

## Evaluation snapshot

AutoRAG uses this model only as a local embedder for retrieval ranking. It is not evaluated as a chat model, and AutoRAG does not republish a training-eval leaderboard. Upstream MTEB results are on the Qwen model card linked below; consult them for benchmark context rather than treating them as AutoRAG guarantees.

## Training-data provenance

AutoRAG does not have independent visibility into the training corpus. Refer to the upstream card:

- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

## License

Apache License, Version 2.0. The notice shipped with AutoRAG is at `licenses/qwen3-embedding-notice.txt`, and the full license text is at `licenses/apache-2.0.txt`.

## Pinned artifact

- Filename: `Qwen3-Embedding-0.6B-Q8_0.gguf`
- Revision: `370f27d7550e0def9b39c1f16d3fbaa13aa67728`
- SHA-256: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`
- URL: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf

## Korean AI Basic Act note

These weights are an embedding model, not a generative model. AutoRAG does not treat them as a generative-AI service that requires chat-output labeling under the Korean AI Basic Act. Operators who wrap AutoRAG's librarian chat model in their own product remain responsible for that model's own labeling duties.
