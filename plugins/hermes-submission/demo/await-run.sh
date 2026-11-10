#!/bin/sh
# Wait for the next Hermes cron run of the automation `autorag-<NAME>` and print the agent's answer.
# Used by demo.tape after a file is dropped into the watched folder; reads only the sandbox Hermes home.
#   usage: await-run.sh [NAME] (default: incoming)
set -eu
NAME="${1:-incoming}"
HH="${DEMO_HERMES_HOME:-/tmp/hh-demo}"
ID=$(python3 -c "
import json, sys
d = json.load(open('$HH/cron/jobs.json')); jobs = d['jobs'] if isinstance(d, dict) else d
print(next(j['id'] for j in jobs if j['name'] == 'autorag-$NAME'))")
OUT="$HH/cron/output/$ID"
before=$(ls "$OUT" 2>/dev/null | wc -l)
printf 'waiting for the autorag-%s cron run...\n' "$NAME"
i=0
while [ "$(ls "$OUT" 2>/dev/null | wc -l)" -le "$before" ]; do
  i=$((i + 1)); [ "$i" -gt 180 ] && { echo "no run after 180 s" >&2; exit 1; }
  sleep 1
done
sleep 2
sed -n '/^## Response/,$p' "$(ls -t "$OUT"/* | head -1)" | sed '1d' | fold -s -w 110
