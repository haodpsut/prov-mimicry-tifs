#!/usr/bin/env bash
# Unzip a node-level (entity) DARPA dataset shipped inside the MAGIC clone.
# These zips are large (theia ~310MB, cadets ~490MB, trace ~1.7GB unpacked), so
# fetch them on demand:  bash setup_entity.sh theia
set -euo pipefail

DATASET="${1:-theia}"
case "$DATASET" in theia|cadets|trace) ;; *) echo "dataset must be theia|cadets|trace"; exit 1;; esac

DIR="MAGIC/data/${DATASET}"
[ -f "${DIR}/graphs.zip" ] || { echo "missing ${DIR}/graphs.zip -- run setup.sh first"; exit 1; }

if [ -f "${DIR}/metadata.json" ]; then
  echo "${DATASET} already unpacked."
else
  echo "unzipping ${DIR}/graphs.zip (this is large) ..."
  unzip -o "${DIR}/graphs.zip" -d "${DIR}/"
fi
echo "files:"; ls -la "${DIR}" | grep -E "metadata|test|train" || true
echo "done -- now: python smoke_attack_entity.py --magic_root ./MAGIC --dataset ${DATASET} --device 0"
