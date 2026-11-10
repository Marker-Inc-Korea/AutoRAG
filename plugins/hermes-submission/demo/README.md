# Demo assets

- `make-fixtures.py`  builds the fictional corpus and the Slack, Telegram, Discord archives (needs `uv`, `slacrawl`, `telecrawl`, `discrawl`)
- `setup-demo.sh`     fixtures + sandbox AutoRAG config + first refresh; refuses a root outside a temp dir
- `telecrawl-preimported.sh`  demo-only wrapper that no-ops `telecrawl import`
- `hermes-demo-env.sh`  runs a command in a scrubbed environment (sandbox HOME, sandbox AutoRAG config)
- `demo.tape`         the VHS script; `demo.mp4` and `demo.gif` are its output
- `await-run.sh`     waits for the next run of the `autorag-incoming` cron job and prints the agent's answer (last scene of the tape)

The last scene (file-system cron) needs, besides the corpus: the `automate` skill's setup run once against the demo
homes with `--path $DEMO_ROOT/incoming --include '**/*.pdf'`, the matching `autorag-incoming` cron job, and
`autorag automate daemon` running under `hermes-demo-env.sh`. See `plugins/hermes/skills/automate/SKILL.md`.

See [`../SUBMISSION.md`](../SUBMISSION.md).
