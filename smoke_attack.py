"""
Smoke test 2 -- insertion-only mimicry attack (proof-of-concept / go-no-go).

Threat model (smoke version): the attacker may ADD benign-looking events (edges)
between existing entities of an attack graph, but may not delete or relabel
anything. We ask: does adding a small number of benign-typed edges pull an attack
graph's MAGIC embedding into the benign manifold, lowering its KNN anomaly score
enough to evade detection at an honestly-calibrated threshold?

Two attack modes:
  random  -- add B benign-typed edges between random existing nodes (no model access)
  greedy  -- at each step, sample C candidate edges, add the one that most lowers
             the anomaly score (white-box-ish: queries the detector's score)

Output: budget curve (mean anomaly score and evasion rate vs B) per mode, plus
the calibrated threshold used. This is a feasibility probe, not the final attack.

Run (from this folder, after setup.sh):
    python smoke_attack.py --magic_root ./MAGIC --dataset streamspot --device 0 \
        --n_graphs 25 --budgets 0 5 10 20 40 --mode both
"""
import argparse
import json
import os
import numpy as np
import torch
import dgl

import common


def benign_edge_type_dist(data, benign_idx, e_feat):
    """Empirical edge-type distribution over benign graphs (what we mimic)."""
    counts = np.zeros(e_feat, dtype=np.float64)
    for i in benign_idx:
        t = data["dataset"][i][0].edata["type"].view(-1).cpu().numpy()
        c = np.bincount(t, minlength=e_feat)
        counts += c
    counts = counts / max(counts.sum(), 1)
    return counts


def add_benign_edges(raw_g, n_add, etype_p, rng):
    """Return a copy of raw_g with n_add benign-typed edges between existing nodes."""
    g = raw_g.clone()
    N = g.num_nodes()
    src = torch.as_tensor(rng.randint(0, N, size=n_add), dtype=torch.long)
    dst = torch.as_tensor(rng.randint(0, N, size=n_add), dtype=torch.long)
    etypes = torch.as_tensor(rng.choice(len(etype_p), size=n_add, p=etype_p), dtype=g.edata["type"].dtype)
    g = dgl.add_edges(g, src, dst, data={"type": etypes})
    return g


def score_of(model, pooler, raw_g, ctx):
    n_feat, e_feat, device, dataset, scorer, _C = ctx
    vec = common.embed_one(model, pooler, raw_g, n_feat, e_feat, device, dataset)[None, :]
    return float(scorer(vec)[0])


def greedy_attack(model, pooler, raw_g, budget, etype_p, rng, ctx):
    """Greedy insertion: each step adds 1 of C candidate edges minimizing the score."""
    _n_feat, _e_feat, _device, _dataset, _scorer, C = ctx
    g = raw_g.clone()
    traj = [score_of(model, pooler, g, ctx)]
    for _ in range(budget):
        best_g, best_s = None, np.inf
        for _c in range(C):
            cand = add_benign_edges(g, 1, etype_p, rng)
            s = score_of(model, pooler, cand, ctx)
            if s < best_s:
                best_s, best_g = s, cand
        g = best_g
        traj.append(best_s)
    return traj  # length budget+1


def random_attack(model, pooler, raw_g, budgets, etype_p, rng, ctx):
    """Add B random benign edges for each B in budgets; return score per budget."""
    out = {}
    for B in budgets:
        g = raw_g if B == 0 else add_benign_edges(raw_g, B, etype_p, rng)
        out[B] = score_of(model, pooler, g, ctx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magic_root", default="./MAGIC")
    ap.add_argument("--dataset", default="streamspot", choices=["streamspot", "wget"])
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--n_graphs", type=int, default=25, help="number of attack graphs to probe")
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 5, 10, 20, 40])
    ap.add_argument("--mode", default="both", choices=["random", "greedy", "both"])
    ap.add_argument("--candidates", type=int, default=32, help="C candidate edges per greedy step")
    ap.add_argument("--alpha", type=float, default=0.01, help="calibrated FPR for the evasion threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/attack_{dataset}.json")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    common.setup_magic(args.magic_root)
    device = torch.device(device if (args.device < 0 or torch.cuda.is_available()) else "cpu")
    rng = np.random.RandomState(args.seed)

    model, pooler, data, n_feat, e_feat = common.load_batch_model(args.dataset, device)
    benign, attack = common.benign_attack_indices(data)

    common._M["set_random_seed"](0)
    X_benign = common.embed_index(model, pooler, data, list(benign), n_feat, e_feat, device, args.dataset)

    # Benign KNN reference + honest calibration threshold (same protocol as smoke_reproduce).
    train_count = 400 if args.dataset == "streamspot" else 100
    n_neighbors = min(int(train_count * 0.02), 10)
    perm = np.random.RandomState(0).permutation(len(benign))
    x_train = X_benign[perm[:train_count]]
    scorer = common.KNNScorer(x_train, n_neighbors)
    cal_scores = scorer(X_benign[perm[train_count:]])
    threshold = float(np.quantile(cal_scores, 1.0 - args.alpha))

    etype_p = benign_edge_type_dist(data, list(benign), e_feat)
    sub = list(attack[: args.n_graphs])
    ctx = (n_feat, e_feat, device, args.dataset, scorer, args.candidates)

    report = {"dataset": args.dataset, "alpha": args.alpha, "threshold": threshold,
              "n_neighbors": n_neighbors, "n_attack_graphs": len(sub), "budgets": args.budgets,
              "note": "evasion = anomaly score falls BELOW the calibrated threshold (misclassified benign)"}

    if args.mode in ("random", "both"):
        per = {B: [] for B in args.budgets}
        for j, i in enumerate(sub):
            res = random_attack(model, pooler, data["dataset"][i][0], args.budgets, etype_p, rng, ctx)
            for B in args.budgets:
                per[B].append(res[B])
            print(f"[random] graph {j+1}/{len(sub)} base={per[args.budgets[0]][-1]:.3f} "
                  f"-> B={args.budgets[-1]}: {res[args.budgets[-1]]:.3f}")
        report["random"] = summarize(per, threshold)

    if args.mode in ("greedy", "both"):
        budget = max(args.budgets)
        trajs = []
        for j, i in enumerate(sub):
            tr = greedy_attack(model, pooler, data["dataset"][i][0], budget, etype_p, rng, ctx)
            trajs.append(tr)
            print(f"[greedy] graph {j+1}/{len(sub)} {tr[0]:.3f} -> {tr[-1]:.3f}")
        trajs = np.array(trajs)  # (n_graphs, budget+1)
        report["greedy"] = {
            "budgets": list(range(budget + 1)),
            "mean_score": trajs.mean(axis=0).tolist(),
            "evasion_rate": (trajs < threshold).mean(axis=0).tolist(),
        }

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out.format(dataset=args.dataset))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k in ("random", "greedy")}, indent=2))
    print(f"\nsaved -> {out_path}")


def summarize(per_budget, threshold):
    out = {"budget": [], "mean_score": [], "evasion_rate": []}
    for B in sorted(per_budget):
        s = np.array(per_budget[B])
        out["budget"].append(B)
        out["mean_score"].append(float(s.mean()))
        out["evasion_rate"].append(float((s < threshold).mean()))
    return out


if __name__ == "__main__":
    main()
