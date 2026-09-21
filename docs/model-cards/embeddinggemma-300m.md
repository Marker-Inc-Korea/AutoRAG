# Model Card: embeddinggemma-300m

This card covers the `embeddinggemma-300m` embedding profile, an optional embedder AutoRAG caches for its loopback-only local gateway. AutoRAG does not train models, does not host a Hugging Face Space, and does not publish user corpora. It fetches one pinned GGUF file and serves embeddings on 127.0.0.1.

## Intended use

Local, loopback-only embedding for AutoRAG's MinSync and native-datasource semantic retrieval. The librarian agent embeds document chunks and queries on the operator's own machine. Corpus text must not be sent to a remote embedding endpoint, and AutoRAG's default configuration never does.

## Unsuitable use

- Chat or text generation of any kind. This model produces vectors, not tokens.
- Treating embedding similarity as ground-truth classification or fact verification.
- Biometric identification, surveillance, or profiling of people.
- Training or fine-tuning a new model from user corpora inside AutoRAG.
- Shipping the weights inside the npm package. Release jobs fetch the pinned URL and verify its SHA-256 instead.
- Substituting a different GGUF build than the pinned artifact below.
- Any use restricted by the Gemma Prohibited Use Policy (see License below).

## Limitations

- This is an embedding model, not a generative or chat model. It cannot answer questions, summarize, or produce text.
- Output dimension is 768 as configured in AutoRAG. The upstream model supports 512, 256, and 128 via Matryoshka truncation, but AutoRAG pins 768.
- This profile requires task prefixes: `task: search result | query: ` for queries and `title: none | text: ` for passages. Embedding text without these prefixes degrades retrieval quality.
- Maximum input context is 2048 tokens per the upstream card. Longer chunks are truncated by AutoRAG's chunking before embedding.
- Per the upstream card, the model was trained on data in 100+ spoken languages. Language-by-language quality varies; the upstream card does not break coverage down per language, and notes the model can struggle with subtle nuance, sarcasm, and figurative language.
- EmbeddingGemma activations do not support float16 inference per the upstream card; AutoRAG runs the pinned Q8_0 GGUF build instead of a float16 path.

## Evaluation snapshot

AutoRAG uses this model only as a local embedder for retrieval ranking. It is not evaluated as a chat model, and AutoRAG does not republish a training-eval leaderboard. Upstream MTEB results, including QAT Q8_0 numbers, are on the Google model cards linked below; consult them for benchmark context rather than treating them as AutoRAG guarantees.

## Training-data provenance

AutoRAG does not have independent visibility into the training corpus. Per the upstream card, the model was trained on roughly 320 billion tokens spanning web documents, code and technical documents, and synthetic task-specific data, with CSAM and sensitive-data filtering applied. Refer to the upstream cards:

- https://ai.google.dev/gemma/docs/embeddinggemma/model_card
- https://huggingface.co/google/embeddinggemma-300m

## License

Gemma Terms of Use (https://ai.google.dev/gemma/terms), including the use restrictions in Section 3.2 and the Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy). The notice shipped with AutoRAG is at `licenses/gemma-notice.txt`, which reproduces the required distribution notice: "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms".

## Pinned artifact

- Filename: `embeddinggemma-300M-Q8_0.gguf`
- Revision: `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12`
- SHA-256: `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63`
- URL: https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf
- GGUF pack: https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF

## Korean AI Basic Act note

These weights are an embedding model, not a generative model. AutoRAG does not treat them as a generative-AI service that requires chat-output labeling under the Korean AI Basic Act. Operators who wrap AutoRAG's librarian chat model in their own product remain responsible for that model's own labeling duties.
