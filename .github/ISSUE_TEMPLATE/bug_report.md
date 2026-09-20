---
name: Bug report (AutoRAG 2.0)
description: Report a defect in the librarian agent, CLI, retrieval methods, or datasource skills
title: "[BUG] "
labels: ["bug", "AutoRAG-2.0"]
---

For defects in the legacy Python AutoRAG pipeline (the `legacy/` directory), use the
[legacy Python bug report](./legacy-bug-report.md) instead.

## Summary

<!-- One sentence: what is broken. -->

## Reproduction

<!-- Exact commands, from a clean starting point where possible. For retrieval
     failures include the configured search paths and the query that fails. -->

```bash

```

## Expected vs actual

<!-- What should have happened, and what happened. -->

## Environment

- AutoRAG version (`autorag --version`, or the git commit):
- Runtime: Bun `bun --version` / Node `node --version`
- OS and architecture: <!-- e.g. macOS 15 arm64, Ubuntu 24.04 x64, Windows 11 -->
- Retrieval involved: <!-- minsync vector / bm25 / hybrid / datasource skill name / none -->
- Workspace state: <!-- e.g. fresh `autorag init`, or an existing .autorag workspace -->

## Diagnostics

<!-- Paste the full error, exit status, and stderr verbatim. AutoRAG reports the
     underlying CLI and filesystem text on purpose, so include it rather than
     summarizing it; that text is what makes the report debuggable. -->

```text

```

For install, configuration, or "it returns nothing" reports, run the `autorag-doctor`
skill and paste its report:

```text

```

## Regression test

- [ ] A failing test that reproduces this is included or linked
- [ ] Reproducible only by hand — steps above are the record
