# Demo assets

- `make-fixtures.py`  builds the fictional corpus and the Slack, Telegram, Discord archives (needs `uv`, `slacrawl`, `telecrawl`, `discrawl`)
- `setup-demo.sh`     fixtures + sandbox AutoRAG config + first refresh; refuses a root outside a temp dir
- `telecrawl-preimported.sh`  demo-only wrapper that no-ops `telecrawl import`
- `hermes-demo-env.sh`  runs a command in a scrubbed environment (sandbox HOME, sandbox AutoRAG config)
- `demo.tape`         the VHS script; `demo.mp4` and `demo.gif` are its output
See [`../SUBMISSION.md`](../SUBMISSION.md).
