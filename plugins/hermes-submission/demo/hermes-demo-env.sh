#!/bin/sh
# Run any command in the scrubbed demo environment: only PATH, a sandbox HOME, the AutoRAG
# sandbox config and the model key survive. Nothing from the real ~/.autorag or ~/.hermes is used.
#
#   DEMO_ROOT    corpus built by make-fixtures.py            (default /tmp/autorag-hermes-demo)
#   DEMO_HERMES_HOME  isolated Hermes home with the plugin enabled (default /tmp/hh-demo)
#   FIREWORKS_API_KEY  model key, forwarded untouched
set -eu
DEMO_ROOT="${DEMO_ROOT:-/tmp/autorag-hermes-demo}"
DEMO_HERMES_HOME="${DEMO_HERMES_HOME:-/tmp/hh-demo}"
exec env -i \
  PATH="${DEMO_PATH:-/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin}" \
  HOME="$DEMO_ROOT/home" HERMES_HOME="$DEMO_HERMES_HOME" TERM="${TERM:-xterm-256color}" LANG=en_US.UTF-8 \
  AUTORAG_HOME="$DEMO_ROOT/autorag-home" AUTORAG_CONFIG="$DEMO_ROOT/autorag-home/config.json" \
  AUTORAG_MCP_READ_ONLY=true FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-}" "$@"
