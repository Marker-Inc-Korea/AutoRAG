SHELL := /bin/bash

# ── Live E2E ────────────────────────────────────────────────────────
# Shared corpus root for live-e2e. Defaults to the current repo path so
# targets pass the checked-out code to the runner. Override with
# E2E_ROOT=</path>; when AUTORAG_LIVE_E2E_ROOT is set it is honored as a
# documented outside-repo default. Targets NEVER create or bootstrap this
# root implicitly — run the explicit bootstrap command first.
E2E_ROOT ?= $(if $(AUTORAG_LIVE_E2E_ROOT),$(AUTORAG_LIVE_E2E_ROOT),$(CURDIR))
QA_IMAGE ?= autorag-qa-linux
QA_PLATFORM ?= linux/amd64
QA_DOCKERFILE ?= scripts/ci/qa.Dockerfile
QA_CONTAINER_HOME ?= /tmp/autorag-home
QA_MODEL_ENV ?= OPENAI_API_KEY AUTORAG_OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY FIREWORKS_API_KEY XAI_API_KEY MISTRAL_API_KEY GROQ_API_KEY AZURE_OPENAI_API_KEY
E2E_EMBEDDER ?= native
E2E_MODE ?= cold

# Recipes read these through make's own environment, never by splicing the
# values into recipe shell source: a value containing $(...) or quotes must
# stay data, not become executable shell code.
export QA_IMAGE QA_PLATFORM QA_DOCKERFILE QA_CONTAINER_HOME QA_MODEL_ENV
export E2E_ROOT E2E_MODE E2E_EMBEDDER E2E_DATASOURCES E2E_ARGS

.PHONY: help install lint format typecheck build test test-all test-macos test-windows test-linux ci supply-chain qa-image qa-shell e2e-live e2e-live-cold e2e-live-docker

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
		'make e2e-live-docker Run live-E2E inside an isolated Docker home' \
		'make qa-shell       Open an isolated Docker shell for manual QA' \
		'make ci            Run lint, typecheck, tests, and build locally' \
		'make supply-chain  Run license, NOTICE, and local CycloneDX gates' \
		'' \
		'  E2E_ROOT=<root>  Shared corpus root (default: current repo path;' \
		'                   honors AUTORAG_LIVE_E2E_ROOT if set)' \
		'  E2E_ARGS=<args>  Extra runner arguments (quoted words supported)' \
		'  E2E_EMBEDDER=native|gateway  Live-E2E embedder (default native)' \
		'  QA_MODEL_ENV=<names>  Model credential allowlist; set empty to' \
		'                   forward none' \
		'  E2E_DATASOURCES  Lane selection (default: local,configured = every lane;' \
		'                   native lanes without a store report SKIP)' \
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

qa-image:
	node scripts/manual-qa/docker-qa.mjs build

qa-shell: qa-image
	node scripts/manual-qa/docker-qa.mjs shell

e2e-live-docker: qa-image
	@test -d "$$E2E_ROOT" || { printf 'E2E_ROOT does not exist: %s\n' "$$E2E_ROOT" >&2; exit 2; }
	node scripts/manual-qa/docker-qa.mjs live

e2e-live:
	E2E_MODE=warm $(MAKE) e2e-live-docker

e2e-live-cold:
	E2E_MODE=cold $(MAKE) e2e-live-docker

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

supply-chain:
	bun scripts/supply-chain/evaluate.ts gate --sbom sbom.local.cdx.json
