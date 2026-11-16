#!/usr/bin/env bash
# CI/release prerequisites from official upstream assets; avoid Ubuntu mirror availability.
set -euo pipefail

tools="${RUNNER_TEMP:?RUNNER_TEMP is required}/autorag-search-tools"
mkdir -p "$tools"
cd "$tools"

curl --fail --location --silent --show-error \
  https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz \
  --output ripgrep.tar.gz
curl --fail --location --silent --show-error \
  https://github.com/sharkdp/fd/releases/download/v10.5.2/fd-v10.5.2-x86_64-unknown-linux-musl.tar.gz \
  --output fd.tar.gz
printf '%s\n' \
  '33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c  ripgrep.tar.gz' \
  '7e2d07fa2b8807642aaec67b7e2972a9fbe4db26fcb340a7372392db42b26694  fd.tar.gz' \
  | sha256sum --check --strict

tar -xzf ripgrep.tar.gz
tar -xzf fd.tar.gz
mkdir -p bin
install -m 755 ripgrep-15.2.0-x86_64-unknown-linux-musl/rg bin/rg
install -m 755 fd-v10.5.2-x86_64-unknown-linux-musl/fd bin/fd
"$tools/bin/rg" --version
"$tools/bin/fd" --version
printf '%s\n' "$tools/bin" >> "${GITHUB_PATH:?GITHUB_PATH is required}"
