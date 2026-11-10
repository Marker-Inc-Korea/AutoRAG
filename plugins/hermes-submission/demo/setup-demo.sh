#!/bin/sh
# Build the isolated demo: synthetic corpus + crawler archives, a sandbox AutoRAG config, first index.
# Touches nothing outside $DEMO_ROOT. Usage: ./setup-demo.sh [DEMO_ROOT]
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO_ROOT="${1:-${DEMO_ROOT:-/tmp/autorag-hermes-demo}}"
case "$DEMO_ROOT" in /tmp/*|/private/tmp/*|"$TMPDIR"*) ;; *) echo "refusing DEMO_ROOT outside a temp dir: $DEMO_ROOT" >&2; exit 1;; esac

uv run "$HERE/make-fixtures.py" "$DEMO_ROOT"
mkdir -p "$DEMO_ROOT/workspace" "$DEMO_ROOT/autorag-home" "$DEMO_ROOT/home/.cache"
# Reuse an already downloaded embedding model instead of fetching ~1.5 GB again.
[ -d "$HOME/.cache/minsync" ] && ln -sfn "$HOME/.cache/minsync" "$DEMO_ROOT/home/.cache/minsync"

export HOME="$DEMO_ROOT/home" AUTORAG_HOME="$DEMO_ROOT/autorag-home" AUTORAG_CONFIG="$DEMO_ROOT/autorag-home/config.json"
case "$AUTORAG_CONFIG" in "$DEMO_ROOT"/*) ;; *) echo "config path escaped the demo root" >&2; exit 1;; esac
AUTORAG="${AUTORAG:-npx -y -p @autorag/librarian@2.13.8 autorag}"

$AUTORAG lite init --search-paths "$DEMO_ROOT/docs" --workspace "$DEMO_ROOT/workspace" \
  --memory-path "$AUTORAG_HOME/memory.json" >/dev/null
python3 - "$HERE" <<'PY'
import json, os, sys
here = sys.argv[1]
p = os.environ["AUTORAG_CONFIG"]
c = json.load(open(p))
c["datasources"] = {
  "slack": {"instanceId": "default", "connector": {"syncSource": "wiretap", "workspace": "T0DEMO01"},
            "description": "Daon Foods Slack: company-wide announcements (#general)."},
  "telegram": {"instanceId": "default", "connector": {"binaryPath": os.path.join(here, "telecrawl-preimported.sh")},
               "description": "Telegram group 'Hanbit x Ops': ops coordination with the Hanbit Logistics account team."},
  "discord": {"type": "discord", "instanceId": "daon", "connector": {"guildId": "G1"},
              "description": "Daon Team Discord, #vendors: vendor contract and legal follow-ups."},
}
json.dump(c, open(p, "w"), indent=2)
PY
$AUTORAG lite refresh --force --json | python3 -c "
import json,sys; d=json.load(sys.stdin); print('refresh ok:', d['ok'], d['counts'])
[print(' ', x['skill'], 'ok' if x['ok'] else 'FAILED') for x in d.get('datasources', [])]"
echo "demo ready: $DEMO_ROOT"
