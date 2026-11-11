#!/bin/sh
# Live E2E for the AutoRAG finder: real Electron + real `hermes serve` (the Hermes checkout's own e2e harness),
# the plugin package from this repo, the real `autorag lite find` CLI, a synthetic corpus. Nothing in the real
# ~/.hermes or ~/.autorag is read or written: the harness sandboxes HERMES_HOME and HOME.
#
#   HERMES_REPO=~/.hermes/hermes-agent LIBRARIAN_CLI=/path/to/librarian/dist/cli/index.js \
#   E2E_ROOT=/tmp/autorag-finder-e2e ./run.sh
#
# E2E_ROOT is built by ../../hermes-submission/demo/setup-demo.sh.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGIN="$(cd "$HERE/../../hermes" && pwd)"
: "${HERMES_REPO:?set HERMES_REPO to a built hermes-agent checkout}"
: "${LIBRARIAN_CLI:?set LIBRARIAN_CLI to a librarian dist/cli/index.js that has `lite find`}"
: "${E2E_ROOT:?set E2E_ROOT to a setup-demo.sh root}"
case "$E2E_ROOT" in /tmp/*|/private/tmp/*) ;; *) echo "E2E_ROOT must be under /tmp" >&2; exit 1;; esac

DESKTOP="$HERMES_REPO/apps/desktop"
cp "$HERE/autorag-finder.spec.ts" "$DESKTOP/e2e/autorag-finder.spec.ts"

# tests-js/scripts import @playwright/test and zod, which pnpm hoists to node_modules/.pnpm/node_modules but does
# not link into the repo root, and ESM resolution ignores NODE_PATH. Link them for this run, remove only what we made.
LINKED=""
HOIST="$HERMES_REPO/node_modules/.pnpm/node_modules"
mkdir -p "$HERMES_REPO/node_modules/@playwright"
for rel in @playwright/test zod; do
  [ -e "$HERMES_REPO/node_modules/$rel" ] || { ln -s "$HOIST/$rel" "$HERMES_REPO/node_modules/$rel"; LINKED="$LINKED $rel"; }
done
cleanup() {
  rm -f "$DESKTOP/e2e/autorag-finder.spec.ts"
  for rel in $LINKED; do rm -f "$HERMES_REPO/node_modules/$rel"; done
  rmdir "$HERMES_REPO/node_modules/@playwright" 2>/dev/null || true
}
trap cleanup EXIT

cd "$DESKTOP"
AUTORAG_E2E_ROOT="$E2E_ROOT" AUTORAG_LIBRARIAN_CLI="$LIBRARIAN_CLI" AUTORAG_PLUGIN_SRC="$PLUGIN" \
  node_modules/.bin/playwright test e2e/autorag-finder.spec.ts --reporter=list --workers=1 "$@"
