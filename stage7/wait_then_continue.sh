#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
while pgrep -f '[p]ython -u -m stage7.run mechanism' >/dev/null; do
    sleep 10
done
COUNT=$(find "$HERE/mechanism" -maxdepth 1 -name '*.json' | wc -l)
if [[ "$COUNT" -ne 9 ]]; then
    echo "Mechanism execution incomplete: $COUNT/9. Formal evaluation remains locked."
    exit 1
fi
exec bash "$HERE/after_mechanism.sh"
