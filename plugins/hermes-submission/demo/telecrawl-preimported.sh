#!/bin/sh
# Demo only: the synthetic Telegram archive is already imported, and `telecrawl import`
# needs a real Telegram Desktop store. Make `import` a no-op; pass everything else through.
for a in "$@"; do
  if [ "$a" = "import" ]; then echo '{"messages":2}'; exit 0; fi
done
exec telecrawl "$@"
