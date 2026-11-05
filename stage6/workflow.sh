#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$HERE/logs"
for command in tests source source-validation adapt freeze formal report; do
  echo "START $command $(date -u +%FT%TZ)"
  bash "$HERE/run.sh" -m stage6.run "$command" "$@" > "$HERE/logs/$command.log" 2>&1
  echo "DONE $command $(date -u +%FT%TZ)"
done
