#!/usr/bin/env bash
set -euo pipefail

bun install --frozen-lockfile
bun run lint
bun run typecheck
bun run test
bun run build
