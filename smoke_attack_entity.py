"""
Smoke test 3 -- node-level insertion-only mimicry (THE go/no-go experiment).

Graph-level StreamSpot is a hard target: the graph embedding is a mean over
~9000 node embeddings, so a sparse edge insertion barely moves it. Node-level
DARPA detection is the natural attack surface: a malicious entity's embedding is
dominated by its small local neighborhood, so adding a few benign in-edges can
pull it into the benign manifold -- and this is exactly where MAGIC's inflated,
test-tuned operating point lives.

Threat model (smoke): for each malicious node v we may ADD benign in-edges
(benign_src -> v) with benign-typed edge features. We never delete/relabel. We
ask: does a small per-node budget B drop v's KNN anomaly score below an honestly
-calibrated threshold (evasion)?

Faithful to MAGIC: same encoder + checkpoints, same KNN-distance score; the
benign KNN reference + the calibration threshold come only from benign TRAIN
nodes (no test access). The whole test graph is re-embedded after insertion so
the GAT message passing is exact.

Defaults target theia (single test graph, smallest). Run on the GPU server:
    python smoke_attack_entity.py --magic_root ./MAGIC --dataset theia --device 0 \
        --n_targets 1000 --budgets 0 2 5 10 20 --seeds 3
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import dgl

import common


def build_benign_reference(model, meta, dataset, device, ref_size, calib_size, rng):
    """Embed benign TRAIN nodes; subsample a KNN reference + a calibration set (faithful, faster)."""
    embs = []
    for i in range(meta["n_train"]):
        g = common.load_entity_graph(dataset, "train", i)
        embs.append(common.embed_entity(model, g, device))
        del g
    X = np.concatenate(embs, axis=0)
    idx = rng.permutation(X.shape[0])
    ref = X[idx[:ref_size]]
    calib = X[idx[ref_size:ref_size + calib_size]]
    return ref, calib


def edge_type_dist(g, e_feat):
    t = g.edata["type"].view(-1).cpu().numpy()
    c = np.bincount(t, minlength=e_feat).astype(np.float64)
    return c / max(c.sum(), 1)


def add_benign_in_edges(g, targets, n_per, benign_pool, etype_p, e_feat, rng):
    """Add n_per benign in-edges (benign_src -> v) to each target node v."""
    if n_per == 0:
        return g
    srcs, dsts, ets = [], [], []
    for v in targets:
        us = benign_pool[rng.randint(0, len(benign_pool), size=n_per)]
        srcs.extend(us.tolist())
        dsts.extend([int(v)] * n_per)
        ets.extend(rng.choice(e_feat, size=n_per, p=etype_p).tolist())
    g2 = g.clone()
    src = torch.as_tensor(srcs, dtype=torch.long)
    dst = torch.as_tensor(dsts, dtype=torch.long)
    et = torch.as_tensor(ets, dtype=g.edata["type"].dtype)
    attr = F.one_hot(et.long(), num_classes=e_feat).float()
    g2 = dgl.add_edges(g2, src, dst, data={"type": et, "attr": attr})
    return g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magic_root", default="./MAGIC")
    ap.add_argument("--dataset", default="theia", choices=["theia", "cadets", "trace"])
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--n_targets", type=int, default=1000, help="malicious nodes to attack (subset for speed)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 2, 5, 10, 20], help="benign in-edges per node")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01, help="calibrated FPR for the evasion threshold")
    ap.add_argument("--ref_size", type=int, default=100000, help="benign KNN reference subsample")
    ap.add_argument("--calib_size", type=int, default=20000, help="benign calibration subsample")
    ap.add_argument("--out", default="results/attack_entity_{dataset}.json")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    common.setup_magic(args.magic_root)
    device = torch.device(device if (args.device < 0 or torch.cuda.is_available()) else "cpu")

    model, meta, n_feat, e_feat = common.load_entity_model(args.dataset, device)
    n_neighbors = 200 if args.dataset == "cadets" else 10   # MAGIC's per-dataset KNN setting
    print(f"[{args.dataset}] n_train={meta['n_train']} n_test={meta['n_test']} "
          f"n_feat={n_feat} e_feat={e_feat} n_neighbors={n_neighbors}")

    if meta["n_test"] != 1:
        raise SystemExit(f"smoke supports single-test-graph datasets (theia/cadets); "
                         f"{args.dataset} has n_test={meta['n_test']} -- handle multi-graph later")

    # malicious node indices (MAGIC stores [indices, names]); single test graph -> direct indices.
    mal = meta["malicious"]
    mal_idx = np.array(mal[0] if isinstance(mal, (list, tuple)) and len(mal) == 2 and
                       isinstance(mal[0], (list, tuple)) else mal, dtype=np.int64)

    rng0 = np.random.RandomState(0)
    ref, calib = build_benign_reference(model, meta, args.dataset, device, args.ref_size, args.calib_size, rng0)
    scorer = common.KNNScorer(ref, n_neighbors)
    threshold = float(np.quantile(scorer(calib), 1.0 - args.alpha))

    g_test = common.load_entity_graph(args.dataset, "test", 0)
    N = g_test.num_nodes()
    etype_p = edge_type_dist(g_test, e_feat)
    is_mal = np.zeros(N, dtype=bool)
    is_mal[mal_idx[mal_idx < N]] = True
    benign_pool = np.where(~is_mal)[0]

    # targets = subset of malicious nodes
    targets_all = mal_idx[mal_idx < N]
    targets = targets_all[: args.n_targets]

    # baseline score of the malicious targets (should sit above threshold = detected)
    base_emb = common.embed_entity(model, g_test, device)
    base_scores = scorer(base_emb[targets])
    base_evasion = float((base_scores < threshold).mean())
    print(f"targets={len(targets)} (of {len(targets_all)} malicious)  threshold={threshold:.3f}  "
          f"baseline mean_score={base_scores.mean():.3f}  baseline evasion={base_evasion:.3f}")

    ev_seeds, sc_seeds = [], []
    for s in range(args.seeds):
        rng = np.random.RandomState(100 + s)
        ev_row, sc_row = [], []
        for B in args.budgets:
            g2 = add_benign_in_edges(g_test, targets, B, benign_pool, etype_p, e_feat, rng)
            emb = common.embed_entity(model, g2, device)
            sc = scorer(emb[targets])
            ev_row.append(float((sc < threshold).mean()))
            sc_row.append(float(sc.mean()))
            del g2, emb
        ev_seeds.append(ev_row)
        sc_seeds.append(sc_row)
        print(f"[entity {args.dataset}] seed {s+1}/{args.seeds} "
              f"evasion {dict(zip(args.budgets, [round(x, 3) for x in ev_row]))}")

    ev, sc = np.array(ev_seeds), np.array(sc_seeds)
    report = {
        "dataset": args.dataset, "level": "node", "alpha": args.alpha, "threshold": threshold,
        "n_neighbors": n_neighbors, "n_targets": int(len(targets)), "n_malicious": int(len(targets_all)),
        "seeds": args.seeds, "budgets": args.budgets,
        "baseline_evasion": base_evasion, "baseline_mean_score": float(base_scores.mean()),
        "evasion_rate_mean": ev.mean(axis=0).tolist(), "evasion_rate_std": ev.std(axis=0).tolist(),
        "mean_score_mean": sc.mean(axis=0).tolist(), "mean_score_std": sc.std(axis=0).tolist(),
        "note": "evasion = malicious node KNN score drops BELOW the calibrated threshold",
    }
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out.format(dataset=args.dataset))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
