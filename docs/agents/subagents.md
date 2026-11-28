---
title: "Host subagents"
description: "How an app that embeds AutoRAG registers its own specialist subagents, and how the librarian delegates to them."
---

# Host subagents

An app that embeds the librarian can hand it specialists that run inside the app's own process, such as a
PDF editor, a spreadsheet builder, or a CRM writer. The librarian delegates a job with one `subagent` tool and
reads back the specialist's summary.

The tool follows pi's `subagent` example extension (`examples/extensions/subagent`):

| Mode | Arguments |
|---|---|
| single | `{ agent, task, files? }` |
| parallel | `{ tasks: [{ agent, task, files? }, …] }` (up to 8, 4 at a time) |
| chain | `{ chain: [{ agent, task, files? }, …] }`, where `{previous}` in a task is the prior step's output |

pi spawns `pi` processes from markdown agent files. A host subagent is instead a `run` function the host
provides, so it can use the host's own engines, windows, and file locks. `files` lists the absolute paths a
task is about.

## Registering

```ts
import { AutoRAGAgent, type SubagentDefinition } from "@autorag/librarian";

const pdfEditor: SubagentDefinition = {
  name: "pdf-editor",
  description: "Edits a PDF in place (annotations, stamps, pages, forms) and checks the result visually. Pass the PDF in `files`.",
  guidance: "Delegate every change to a PDF's content or pages to pdf-editor; never edit PDFs with shell tools.",
  mutatesFiles: true,
  async run({ task, files }, { signal, onUpdate }) {
    onUpdate?.("editing…");
    const summary = await myPdfAgent.edit(files[0], task, signal);
    return { output: summary, changedFiles: [files[0]] };
  },
};

const agent = new AutoRAGAgent({ /* … */ subagents: [pdfEditor] });
```

`name` is 2–40 lowercase letters, digits, or dashes. Names must be unique.

## Prompt caching

The `subagent` tool and a `## Subagents` system-prompt section are built once per agent from the
registered definitions, sorted by name. They contain nothing that changes between turns, so they do
not break the provider's prompt cache. Keep `description` and `guidance` static: put per-request details
(which file is open, what the user selected) in the task or the conversation, never in the definition.

With no subagents registered nothing is added: the CLI and every host that registers none get exactly the
same prompt and tools as before.

## Permissions

A subagent that writes files sets `mutatesFiles: true`. In `readonly` [permission mode](./permissions.md) a
`subagent` call naming such an agent is blocked before it runs, the same way `write` and `edit` are blocked.
Subagents without `mutatesFiles` run in both modes.
