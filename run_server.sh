#!/usr/bin/env bash
# Run the smoke test on the GPU server inside a tmux session using a conda env,
# so it survives SSH disconnects. Override via env vars, e.g.:
#   ENV=prov-mimicry SESSION=prov-smoke DATASET=streamspot DEVICE=0 CUDA=cu118 bash run_server.sh
set -euo pipefail

ENV="${ENV:-prov-mimicry}"
SESSION="${SESSION:-prov-smoke}"
DATASET="${DATASET:-streamspot}"
DEVICE="${DEVICE:-0}"
CUDA="${CUDA:-cu121}"               # RTX 4090 server runs CUDA 12.1
TORCH="${TORCH:-2.1.0}"            # torch version (cu121 build)
TORCH_CH="${TORCH_CH:-torch-2.1}"  # DGL wheel channel matching the torch version
RECREATE="${RECREATE:-0}"          # set 1 to wipe and rebuild a broken env

command -v conda >/dev/null || { echo "conda not found on PATH"; exit 1; }
command -v tmux  >/dev/null || { echo "tmux not found on PATH"; exit 1; }

if [ "${RECREATE}" = "1" ]; then
  echo ">> removing existing env '${ENV}'"
  conda env remove -n "${ENV}" -y || true
fi

# 1) create the conda env once (CPU deps), then add torch + dgl for the server CUDA.
if ! conda env list | grep -qE "/${ENV}\$"; then
  echo ">> creating conda env '${ENV}'  (CUDA=${CUDA}, torch=${TORCH})"
  conda env create -f environment.yml -n "${ENV}"
  conda run -n "${ENV}" pip install "torch==${TORCH}" --index-url "https://download.pytorch.org/whl/${CUDA}"
  conda run -n "${ENV}" pip install dgl -f "https://data.dgl.ai/wheels/${TORCH_CH}/${CUDA}/repo.html"
fi

# verify the stack imports + sees the GPU before doing real work
conda run -n "${ENV}" python - <<'PY'
import torch, dgl
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| cuda available", torch.cuda.is_available())
print("dgl", dgl.__version__, "imports OK")
PY

mkdir -p results

# 2) launch everything in a detached tmux session.
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "
  source \"\$(conda info --base)/etc/profile.d/conda.sh\";
  conda activate ${ENV};
  python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())';
  bash setup.sh;
  python smoke_reproduce.py --magic_root ./MAGIC --dataset ${DATASET} --device ${DEVICE} 2>&1 | tee results/log_reproduce_${DATASET}.txt;
  python smoke_attack.py    --magic_root ./MAGIC --dataset ${DATASET} --device ${DEVICE} --mode both 2>&1 | tee results/log_attack_${DATASET}.txt;
  echo '=== DONE. commit results:  git add -f results/*  && git commit -m \"server smoke results\" && git push ===';
  exec bash
"
echo "Launched tmux session '${SESSION}'. Attach with:  tmux attach -t ${SESSION}"
echo "Detach inside tmux with:  Ctrl-b then d"
