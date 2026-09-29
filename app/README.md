# AutoRAG Finder (desktop app)

Electron + React + TypeScript desktop shell for the AutoRAG Agent — the AI Finder
main window. The design contract lives in [DESIGN.md](DESIGN.md); the reference
prototype and handoff README live in `../docs/design-reference/ai-finder/`.

## Prerequisites

| Requirement | Why | Install |
|---|---|---|
| Bun | package manager and test runner | https://bun.sh |
| Node.js 24+ | Electron tooling | https://nodejs.org |
| **dupey CLI** | **required** — version stacks (duplicate/near-duplicate families) and exact-duplicate exclusion; the app reports an error banner without it | `cargo install dupey --locked` (needs the Rust toolchain: https://rustup.rs) |

`make install` from the repository root provisions dupey automatically (it runs
`cargo install dupey --locked` when dupey is missing) and fails loudly when cargo
is unavailable. `bun run dev` and `bun run build` in this directory verify dupey
through `scripts/require-dupey.mjs` and refuse to run without it.

`AUTORAG_SKIP_DUPEY_CHECK=1` bypasses the check for environments that
intentionally run without dupey; the app then shows the visible
`.list__error` banner instead of version stacks. `AUTORAG_SKIP_DUPEY=1` does the
same for `make install`.

## Run

From the repository root:

```bash
make electron PORT=9234        # dev app (Make variable, not --port)
```

Or in this directory: `PORT=9234 bun run dev`.

## Verify

```bash
bun run typecheck    # tsc, node + web projects
bun run test         # vitest
bun run build        # electron-vite build (runs the library build first)
```
