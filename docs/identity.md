# Personal context with IDENTITY.md

The answering librarian can use a file you maintain at
`<AUTORAG_HOME>/IDENTITY.md`, or `~/.autorag/IDENTITY.md` when `AUTORAG_HOME` is
unset. For example:

```markdown
My name is Alex. I work as a research librarian.
I prefer answers in English unless I ask for another language.
```

AutoRAG reads this UTF-8 file once when it constructs an answering Agent and adds
nonblank content verbatim to a labeled section of the system prompt. Missing or
whitespace-only files leave the existing prompt unchanged. AutoRAG does not create,
edit, validate, or truncate the file. Other read errors are reported, so an
unreadable file does not silently appear to have loaded.

Restart AutoRAG (or construct a new `AutoRAGAgent`) after editing or deleting the
file. Changing models, starting a new conversation, or reloading an existing Agent
does not reread it. Selecting another config with `--config` or `AUTORAG_CONFIG`,
changing the workspace, or choosing another memory file does not change the
identity location; use `AUTORAG_HOME` to select a different home.

The Discord, Telegram, and WhatsApp bridged CLI Agents skip identity loading for
the entire shared Agent, including its terminal conversation. Model-free
refresh/status/watch/index operations, `AutoRAGLite`, and profile retrieval-tool
backing Agents also skip it. Standalone profile sessions retain their own prompts.
SDK hosts can pass `loadIdentity: false` when constructing an Agent, including
custom bridge hosts. This is a runtime option, not a config-file profile schema.
Skipping automatic injection does not restrict a bridge's existing file-access tools.

Identity text is sent to the configured model as system context and consumes its
context window. Existing model/context limits still apply; keep the file concise.
It is personal background, not a change to retrieval ranking or evidence rules.

Pi's existing session storage records system prompts, including identity text.
A newly constructed Agent uses the current file when resuming a session, but
deleting the file does not erase old session records or previous model replies.
