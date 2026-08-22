SHELL := /bin/bash

.PHONY: help install lint format typecheck build test test-all test-macos test-windows test-linux ci

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
		'make test-windows  Run the Windows-compatible suite' \
		'make test-linux    Run portable Linux CI in a Docker container' \
		'make ci            Run lint, typecheck, tests, and build locally'

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

test-macos:
	@test "$$(uname -s)" = "Darwin" || { echo "test-macos requires a macOS host"; exit 1; }
	bun run test

test-windows:
	@case "$$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ;; *) echo "test-windows must run from Git Bash/MSYS2 on Windows"; exit 1;; esac
	bun run test:windows

test-linux:
	docker build --platform linux/amd64 -f scripts/ci/linux.Dockerfile -t autorag-ci-linux-amd64 scripts/ci
	docker run --rm --platform linux/amd64 -v "$$(pwd):/workspace" -v /workspace/node_modules -w /workspace autorag-ci-linux-amd64

ci: lint typecheck test build
