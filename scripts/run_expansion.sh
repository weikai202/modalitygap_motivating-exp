#!/usr/bin/env bash
set -euo pipefail
expansion_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for expansion_dataset in crisismmd hatefulmemes newsclippings; do
  bash "$expansion_root/scripts/run_experiment.sh" \
    --config "$expansion_root/configs/expansion-${expansion_dataset}-8epoch-v1.json" "$@"
done
