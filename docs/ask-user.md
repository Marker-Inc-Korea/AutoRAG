# Asking you questions (`ask_user`)

In `autorag tui` the librarian can pause and ask you a question instead of guessing, the way Claude Code's
AskUserQuestion does. It does this when a request is ambiguous or a decision is yours (which report, which
folder, keep or delete).

The tool is `ask_user` from the pinned [`pi-ask-user`](https://www.npmjs.com/package/pi-ask-user) extension
(`0.17.0`, MIT), loaded into the interactive session by `src/agent/ask-user.ts`.

## What you see

- **Multiple choice**: a searchable list with a details pane. `↑↓` move, `Enter` selects, type to filter.
  A question can allow several picks (`allowMultiple`) and a free-text answer (`allowFreeform`, on by default).
- **Open question**: a text prompt.
- **Batch**: 2-4 independent questions shown together.
- `Esc` cancels. The model is told `User cancelled the question` and carries on without inventing an answer.
- `alt+o` hides and shows the popup while you check something else.

The model receives your answer as `User answered: <choice or text>`.

## When it is offered

Only when you are at the terminal. A session a chat bridge drives (`--telegram`, `--discord`, `--whatsapp`,
or any extra pi extension path) has nobody to answer a terminal dialog, so the tool is not registered there
and the system prompt does not mention it. Discord brings its own question tool.

Priority the model is given (system prompt and phase prompts): search and explore first with the other tools,
use `ask_user` only for what still cannot be verified or a decision that is the user's alone, and when it must
ask the user anything, ask through the tool instead of in plain text. The two-phase search's "no tools"
instruction for the fast and direct answers does not cover `ask_user`; those prompts name it as the one
exception under the same rule.

## Settings

`pi-ask-user` reads `~/.pi/agent/ask-user.json` (display mode, single-select layout, shortcuts) and the
`PI_ASK_USER_*` environment variables. See the package README for the keys.

## Tests

- `test/agent/ask-user.test.ts`: offline, through the real interactive session with a scripted model and UI.
- `test/live-e2e/ask-user.test.ts`: a real model decides to ask. Gated:
  `AUTORAG_ASK_USER_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/ask-user.test.ts`
