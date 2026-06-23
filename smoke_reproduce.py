"""
Smoke test 1 -- reproduce MAGIC and quantify the threshold-tuning inflation.

Two questions:
 (A) Can we reproduce MAGIC's reported headline (test-tuned threshold, max-F1 on
     the test labels, as in MAGIC's own evaluate_batch_level_using_knn)?
 (B) What happens under a *deployment-realistic* protocol where the threshold is
     fixed on a held-out BENIGN calibration split (no test access), then applied
     blind to the test set?

The gap between (A) and (B) is the inflation we flagged. This script does not
attack anything; it establishes the honest baseline the rest of the project
measures against.

Run (from this folder, after setup.sh):
    python smoke_reproduce.py --magic_root ./MAGIC --dataset streamspot --device 0
"""
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

import common


def test_tuned_metrics(scores, y):
    """MAGIC-style: pick the threshold that maximizes F1 ON THE TEST LABELS."""
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    k = int(np.argmax(f1))
    t = thr[min(k, len(thr) - 1)]
    pred = (scores >= t).astype(int)
    return dict(threshold=float(t), precision=float(prec[k]), recall=float(rec[k]),
                f1=float(f1[k]), fpr=_fpr(y, pred), auc=float(roc_auc_score(y, scores)))


def calibrated_metrics(scores, y, cal_scores, alpha):
    """Deployment-realistic: threshold = (1-alpha) quantile of BENIGN calibration scores."""
    t = float(np.quantile(cal_scores, 1.0 - alpha))
    pred = (scores >= t).astype(int)
    return dict(alpha=alpha, threshold=t, precision=_prec(y, pred), recall=_recall(y, pred),
                f1=_f1(y, pred), fpr=_fpr(y, pred), auc=float(roc_auc_score(y, scores)))


def _fpr(y, p):
    neg = (y == 0)
    return float((p[neg] == 1).sum() / max(neg.sum(), 1))


def _recall(y, p):
    pos = (y == 1)
    return float((p[pos] == 1).sum() / max(pos.sum(), 1))


def _prec(y, p):
    pp = (p == 1)
    return float((y[pp] == 1).sum() / max(pp.sum(), 1))


def _f1(y, p):
    r, pr = _recall(y, p), _prec(y, p)
    return float(2 * pr * r / (pr + r + 1e-9))


def _agg(dicts, keys):
    """mean/std across a list of metric dicts, for the given keys."""
    out = {}
    for k in keys:
        v = np.array([d[k] for d in dicts], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magic_root", default="./MAGIC")
    ap.add_argument("--dataset", default="streamspot", choices=["streamspot", "wget"])
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--seeds", type=int, default=5, help="number of split seeds to average (Trans-grade)")
    ap.add_argument("--out", default="results/reproduce_{dataset}.json")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    common.setup_magic(args.magic_root)
    import torch
    device = torch.device(device if (args.device < 0 or torch.cuda.is_available()) else "cpu")

    model, pooler, data, n_feat, e_feat = common.load_batch_model(args.dataset, device)
    benign, attack = common.benign_attack_indices(data)
    print(f"[{args.dataset}] benign={len(benign)} attack={len(attack)} n_feat={n_feat} e_feat={e_feat}")

    # Embed everything once (deterministic, label-free).
    common._M["set_random_seed"](0)
    X_benign = common.embed_index(model, pooler, data, list(benign), n_feat, e_feat, device, args.dataset)
    X_attack = common.embed_index(model, pooler, data, list(attack), n_feat, e_feat, device, args.dataset)

    train_count = 400 if args.dataset == "streamspot" else 100
    n_neighbors = min(int(train_count * 0.02), 10)
    alphas = (0.001, 0.01, 0.05)

    # Repeat over split seeds: embeddings are fixed; only the benign train/calib/test
    # partition (and hence the KNN reference + honest threshold) varies per seed.
    tuned_runs, calib_runs = [], {a: [] for a in alphas}
    for s in range(args.seeds):
        perm = np.random.RandomState(s).permutation(len(benign))
        tr, rest = perm[:train_count], perm[train_count:]
        cal, ben_test = rest[: len(rest) // 2], rest[len(rest) // 2:]
        x_train = X_benign[tr]
        x_eval = np.concatenate([X_benign[ben_test], X_attack], axis=0)
        y_eval = np.concatenate([np.zeros(len(ben_test)), np.ones(len(X_attack))])
        scorer = common.KNNScorer(x_train, n_neighbors)
        cal_scores, eval_scores = scorer(X_benign[cal]), scorer(x_eval)
        tuned_runs.append(test_tuned_metrics(eval_scores, y_eval))
        for a in alphas:
            calib_runs[a].append(calibrated_metrics(eval_scores, y_eval, cal_scores, a))

    mkeys = ["precision", "recall", "f1", "fpr", "auc"]
    report = {
        "dataset": args.dataset, "seeds": args.seeds, "n_neighbors": n_neighbors,
        "split": {"train": int(train_count), "attack_test": int(len(X_attack))},
        "A_test_tuned (MAGIC-style, leaks test labels)": _agg(tuned_runs, mkeys),
        "B_calibrated (honest, no test access)": {
            f"alpha={a}": _agg(calib_runs[a], mkeys) for a in alphas
        },
    }
    import os
    # setup_magic() chdir'd into MAGIC, so anchor outputs to this file's folder.
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out.format(dataset=args.dataset))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
