---
title: "Asking you questions"
description: "The ask_user tool the librarian uses to pause for a decision, when it is offered, and how to configure it."
---

# Asking you questions

In `autorag tui` the librarian can pause and ask you a question instead of guessing. It does this
when a request is ambiguous or a decision is yours (which report, which folder, keep or delete).

The tool is `ask_user`, registered by the pinned [`pi-ask-user`](https://www.npmjs.com/package/pi-ask-user)
extension (`0.17.0`, MIT), loaded into the interactive session by `src/agent/ask-user.ts`.

## What you see

- **Multiple choice**: a searchable list with a details pane. `↑`/`↓` move, `Enter` selects, typing
  filters. A question can allow several picks (`allowMultiple`) and a free-text answer
  (`allowFreeform`, on by default).
- **Open question**: a text prompt.
- **Batch**: 2–4 independent questions shown together with a review page before submitting.
- `Esc` cancels. The model is told `User cancelled the question` and carries on without inventing an
  answer.
- `alt+o` hides and shows the popup while you check something else (overlay mode only).

The model receives your answer as `User answered: <choice or text>`; a batch reports
`User answered N of M questions:`.

## When it is offered

Only when you are at the terminal. A session a chat bridge drives (`--telegram`, `--discord`,
`--whatsapp`, or any extra Pi extension path) has nobody to answer a terminal dialog, so the tool is
not registered there and the system prompt does not mention it. Discord brings its own question tool
(`discord_ask_user_question`).

The model's priority is: search and explore first with the other tools, use `ask_user` only for what
still cannot be verified or a decision that is the user's alone, and whenever it must ask the user
anything, ask through the tool instead of in plain text. The fast answer's rule against local
search, retrieval, datasource, and file-reading tools does not cover `ask_user` (nor the calculator
or web tools); that prompt names the exceptions under the same rule.

## Settings

Per-call tool parameters override both settings sources. Environment variables win over the settings
file.

Create `~/.pi/agent/ask-user.json` (or `<agent-dir>/ask-user.json` when `PI_CODING_AGENT_DIR` is
set):

```json
{
  "displayMode": "inline",
  "singleSelectLayout": "list",
  "allowComment": false,
  "contextExpanded": false,
  "overlayToggleKey": "alt+o",
  "commentToggleKey": "ctrl+g",
  "emitFullEvents": false
}
```

All keys are optional; unknown keys and wrong-typed values are ignored. The file is read on every
`ask_user` call, so edits apply without restarting. A file that exists but is not valid JSON fails
the call with the path in the error.

The matching environment variables are `PI_ASK_USER_DISPLAY_MODE`, `PI_ASK_USER_SINGLE_SELECT_LAYOUT`,
`PI_ASK_USER_ALLOW_COMMENT`, `PI_ASK_USER_OVERLAY_TOGGLE_KEY`, `PI_ASK_USER_COMMENT_TOGGLE_KEY`,
`PI_ASK_USER_CONTEXT_EXPANDED`, and `PI_ASK_USER_EMIT_FULL_EVENTS`. They must be present in the
process that launches the TUI.

::: warning **Warning:** `emitFullEvents` (or `PI_ASK_USER_EMIT_FULL_EVENTS=true`) shares answers
and prompt context with other installed extensions through lifecycle events, including selections,
comments, and freeform text. Leave it `false` unless an extension you trust needs that data.
:::

See the [pi-ask-user README](https://www.npmjs.com/package/pi-ask-user) for the full parameter list,
display modes, and keybindings.
