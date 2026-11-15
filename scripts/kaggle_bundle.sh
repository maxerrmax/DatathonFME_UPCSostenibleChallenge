#!/usr/bin/env bash
# Empaqueta el codi de l'arena + dades processades + models dels plecs com a dataset PRIVAT de Kaggle.
#   bash scripts/kaggle_bundle.sh create   # primera vegada
#   bash scripts/kaggle_bundle.sh          # nova versió del dataset
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_KG="${KAGGLE_USER:-jordixu}"
STAGE="${STAGE:-$ROOT/.kaggle_stage}"
rm -rf "$STAGE"; mkdir -p "$STAGE/repo"
cd "$ROOT"
# codi (sense pkl antics, sense resultats) + models dels plecs + dades processades
tar --exclude='__pycache__' --exclude='dificil/arena/results' --exclude='*.zip' \
    --exclude='dificil/agent_hard/models_all.pkl' \
    -cf - dificil estandard/data.py starter_kit/environment.py starter_kit/battery_spec.json \
          scripts/arena.py data/processed/hard_hourly.csv data/external/ree_hourly.csv | tar -xf - -C "$STAGE/repo"
(cd "$STAGE" && tar -czf bundle.tgz repo && rm -rf repo)
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "jamm-arena-bundle", "id": "$USER_KG/jamm-arena-bundle", "licenses": [{"name": "other"}]}
EOF
ls -la "$STAGE"
if [ "${1:-}" = "create" ]; then
  (cd "$STAGE" && kaggle datasets create -p .)
else
  (cd "$STAGE" && kaggle datasets version -p . -m "update $(date +%H:%M)")
fi
