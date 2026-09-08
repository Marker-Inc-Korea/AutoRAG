SHELL := /bin/bash

# ── Live E2E ────────────────────────────────────────────────────────
# Shared corpus root for live-e2e. Defaults to the current repo path so
# targets pass the checked-out code to the runner. Override with
# E2E_ROOT=</path>; when AUTORAG_LIVE_E2E_ROOT is set it is honored as a
# documented outside-repo default. Targets NEVER create or bootstrap this
# root implicitly — run the explicit bootstrap command first.
E2E_ROOT ?= $(if $(AUTORAG_LIVE_E2E_ROOT),$(AUTORAG_LIVE_E2E_ROOT),$(CURDIR))

.PHONY: help install lint format typecheck build test test-all test-macos test-windows test-linux ci e2e-live e2e-live-cold

help:
	@printf '%s\n' \
		'make install       Install dependencies from bun.lock' \
		'make lint          Check formatting and lint rules' \
		'make format        Apply Biome formatting and safe fixes' \
		'make typecheck     Run TypeScript type checking' \
		'make build         Build library, CLI, and declarations' \
		'make test          Run the complete AutoRAG 2.0 test suite' \
		'make test-all      Alias for the complete test suite' \
		'make test-macos    Run the complete suite on a macOS host' \
		'make test-windows  Run the complete suite on a Windows host' \
		'make test-linux    Run the complete suite in a Linux container' \
		'make e2e-live      Run warm live-E2E (reuse clone-local state)' \
		'make e2e-live-cold Run cold live-E2E (delete state, rebuild)' \
		'make ci            Run lint, typecheck, tests, and build locally' \
		'' \
		'  E2E_ROOT=<root>  Shared corpus root (default: current repo path;' \
		'                   honors AUTORAG_LIVE_E2E_ROOT if set)' \
		'  E2E_ARGS=<args>  Extra arguments forwarded to the runner' \
		'  bootstrap first: node scripts/live-e2e/runner.mjs bootstrap --root "$$AUTORAG_LIVE_E2E_ROOT"'

install:
	bun install --frozen-lockfile

lint:
	bun run lint

format:
	bun run check

typecheck:
	bun run typecheck

build:
	bun run build

test:
	bun run test

test-all: test

e2e-live:
	E2E_DATASOURCES="$(E2E_DATASOURCES)" node scripts/live-e2e/runner.mjs live --mode warm --root "$(E2E_ROOT)" $(E2E_ARGS)

e2e-live-cold:
	E2E_DATASOURCES="$(E2E_DATASOURCES)" node scripts/live-e2e/runner.mjs live --mode cold --root "$(E2E_ROOT)" $(E2E_ARGS)

test-macos:
	@test "$$(uname -s)" = "Darwin" || { echo "test-macos requires a macOS host"; exit 1; }
	bun run test

test-windows:
	@case "$$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ;; *) echo "test-windows must run from Git Bash/MSYS2 on Windows"; exit 1;; esac
	bun run test

test-linux:
	docker build --platform linux/amd64 -f scripts/ci/linux.Dockerfile -t autorag-ci-linux-amd64 scripts/ci
	docker run --rm --platform linux/amd64 -v "$$(pwd):/workspace" -v /workspace/node_modules -w /workspace autorag-ci-linux-amd64

ci: lint typecheck test build
