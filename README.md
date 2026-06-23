# prov-evasion (smoke test)

Feasibility probe for a TIFS paper: **insertion-only mimicry evasion + calibrated
defense against self-supervised provenance-graph APT detectors** (MAGIC class).

This stage answers two de-risking questions before we formalize anything:

1. **Inflation** — how much does the de-facto evaluation protocol (KNN-distance +
   threshold tuned on the *test* labels) over-report vs an honest threshold fixed
   on a benign calibration split with no test access? (`smoke_reproduce.py`)
2. **Attackability** — does adding a handful of benign-looking edges to an attack
   graph pull its MAGIC embedding into the benign manifold and evade detection at
   an honestly-calibrated threshold? (`smoke_attack.py`)

We build only on the **public** MAGIC repo (`FDUDSDE/MAGIC`), which ships model
checkpoints and pre-processed StreamSpot/Wget graphs, so the smoke test attacks
the *published* detector and needs no large DARPA download.

## Run on the GPU server (conda + tmux)

One command creates the conda env (adding a 4090-compatible torch+dgl) and runs
the whole smoke test inside a detached **tmux** session so it survives SSH drops:

```bash
# adjust CUDA tag to the server (cu118 / cu121); DATASET/DEVICE optional
CUDA=cu118 DATASET=streamspot DEVICE=0 bash run_server.sh
tmux attach -t prov-smoke      # watch it; detach with Ctrl-b then d
```

Or step by step:

```bash
conda env create -f environment.yml -n prov-mimicry && conda activate prov-mimicry
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
bash setup.sh                      # clones MAGIC + unzips its graphs

python smoke_reproduce.py --magic_root ./MAGIC --dataset streamspot --device 0 --seeds 5
python smoke_attack.py    --magic_root ./MAGIC --dataset streamspot --device 0 \
    --n_graphs 25 --budgets 0 5 10 20 40 --mode both --candidates 32 --seeds 3
```

Both scripts average over seeds and report mean/std (Trans-grade): `--seeds` is the
number of split seeds (reproduce) / attack seeds (attack). For a first quick probe
use `--seeds 1`; scale up once the GPU run is confirmed working.

Results (JSON + tee'd logs) land in `results/`. **Please commit and push back**:
`git add -f results/* && git commit -m "server smoke results" && git push`
(`.gitignore` skips the big `MAGIC/` clone and the pkl, so force-add `results/`).

## What to look for (go / no-go)

- `smoke_reproduce.py`: a large gap between `A_test_tuned` and `B_calibrated`
  (recall/F1 collapsing once the threshold can't peek at test) confirms the
  inflation thesis.
- `smoke_attack.py`: `evasion_rate` rising with budget `B`, especially for
  `greedy`, at a small `B` relative to graph size (StreamSpot graphs have
  thousands of edges) confirms the attack is feasible -> green light.

## Files
- `common.py` — faithful MAGIC model/data loading + KNN scorer.
- `smoke_reproduce.py` — reproduce vs honest-calibrated metrics.
- `smoke_attack.py` — insertion-only mimicry (random + greedy) budget curves.

This is a feasibility probe, **not** the final attack/defense. Threat model,
adaptive defense, and entity-level (DARPA) experiments come after go/no-go.
