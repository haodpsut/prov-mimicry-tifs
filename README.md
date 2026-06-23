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

## Run on the GPU server

```bash
bash setup.sh                      # clones MAGIC + unzips its graphs
pip install -r requirements.txt    # see notes re: RTX 4090 / CUDA in the file

# 1) reproduce + inflation gap
python smoke_reproduce.py --magic_root ./MAGIC --dataset streamspot --device 0

# 2) insertion mimicry POC (random + greedy)
python smoke_attack.py --magic_root ./MAGIC --dataset streamspot --device 0 \
    --n_graphs 25 --budgets 0 5 10 20 40 --mode both --candidates 32
```

Results are written to `results/*.json`. **Please commit `results/` and push back**
(the `.gitignore` ignores the big `MAGIC/` clone and the pkl, but `git add -f
results/*.json` to include the outputs).

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
