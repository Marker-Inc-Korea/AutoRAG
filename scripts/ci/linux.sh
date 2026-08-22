#!/usr/bin/env bash
set -euo pipefail

bun install --frozen-lockfile
bun run lint
bun run typecheck
bun run test -- \
	test/smoke.test.ts \
	test/manifest \
	test/memory \
	test/parser \
	test/process \
	test/retrieval \
	test/security \
	test/types
bun run build
