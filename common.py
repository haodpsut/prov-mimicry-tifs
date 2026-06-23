"""
Shared helpers for the provenance-evasion smoke test.

Everything here loads the *public* MAGIC artifacts (model + shipped checkpoints +
shipped pre-processed StreamSpot/Wget graphs) and reproduces MAGIC's own
embedding + KNN scoring path faithfully, so that any later attack/defense result
is measured against the real published detector, not a re-implementation.

Usage: scripts call setup_magic(magic_root) once, which chdir's into the MAGIC
checkout (MAGIC uses relative ./data and ./checkpoints paths) and imports its
modules.
"""
import os
import sys
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# Filled in by setup_magic()
_M = {}


def setup_magic(magic_root):
    """chdir into a MAGIC checkout and import its modules. Returns a dict of refs."""
    magic_root = os.path.abspath(magic_root)
    if not os.path.isdir(magic_root):
        raise FileNotFoundError(f"MAGIC root not found: {magic_root} (run setup.sh first)")
    os.chdir(magic_root)
    if magic_root not in sys.path:
        sys.path.insert(0, magic_root)
    from model.autoencoder import build_model          # noqa: E402
    from utils.loaddata import load_batch_level_dataset, transform_graph  # noqa: E402
    from utils.poolers import Pooling                   # noqa: E402
    from utils.utils import set_random_seed             # noqa: E402
    _M.update(dict(build_model=build_model,
                   load_batch_level_dataset=load_batch_level_dataset,
                   transform_graph=transform_graph,
                   Pooling=Pooling,
                   set_random_seed=set_random_seed,
                   root=magic_root))
    return _M


class Args:
    """Minimal stand-in for MAGIC's argparse namespace (only what build_model reads)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def load_batch_model(dataset, device):
    """Load a batch-level (streamspot/wget) MAGIC detector + its dataset. Faithful to MAGIC root eval.py."""
    assert dataset in ("streamspot", "wget")
    _M["set_random_seed"](0)
    data = _M["load_batch_level_dataset"](dataset)
    n_feat, e_feat = data["n_feat"], data["e_feat"]
    args = Args(num_hidden=256, num_layers=4, negative_slope=0.2, mask_rate=0.5,
                alpha_l=3, n_dim=n_feat, e_dim=e_feat, pooling="mean")
    model = _M["build_model"](args)
    ckpt = f"./checkpoints/checkpoint-{dataset}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    pooler = _M["Pooling"]("mean")
    return model, pooler, data, n_feat, e_feat


@torch.no_grad()
def embed_one(model, pooler, raw_g, n_feat, e_feat, device, dataset):
    """MAGIC graph embedding for a single raw DGL graph (ndata['type'], edata['type'])."""
    g = _M["transform_graph"](raw_g, n_feat, e_feat).to(device)
    out = model.embed(g)
    if dataset == "wget":
        vec = pooler(g, out, n_types=n_feat)
    else:
        vec = pooler(g, out)
    return vec.squeeze(0).detach().cpu().numpy()


@torch.no_grad()
def embed_index(model, pooler, data, idxs, n_feat, e_feat, device, dataset):
    """Embed a list of dataset indices -> (len, D) array."""
    vecs = []
    for i in idxs:
        raw_g = data["dataset"][i][0]
        vecs.append(embed_one(model, pooler, raw_g, n_feat, e_feat, device, dataset))
    return np.stack(vecs, axis=0)


class KNNScorer:
    """MAGIC's KNN-distance anomaly scorer, prebuilt once from benign training data.

    score(e) = mean_kNN_dist(e) / mean_kNN_dist(train).  Higher = more anomalous.
    The KNN reference and standardization stats use ONLY benign training data --
    they never see eval labels. (The leakage in MAGIC/FINE is in how the *threshold*
    is later picked, not here; see smoke_reproduce.py.) Reusing a fitted scorer
    matters because the greedy attack queries it tens of thousands of times.
    """
    def __init__(self, x_train, n_neighbors):
        self.mu = x_train.mean(axis=0)
        self.sd = x_train.std(axis=0) + 1e-6
        xtr = (x_train - self.mu) / self.sd
        self.k = n_neighbors
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(xtr)
        d_tr, _ = self.nbrs.kneighbors(xtr, n_neighbors=n_neighbors)
        self.mean_distance = d_tr.mean() * n_neighbors / (n_neighbors - 1)

    def __call__(self, x_eval):
        xev = (np.atleast_2d(x_eval) - self.mu) / self.sd
        d_ev, _ = self.nbrs.kneighbors(xev, n_neighbors=self.k)
        return d_ev.mean(axis=1) / self.mean_distance


def knn_anomaly_scores(x_train, x_eval, n_neighbors):
    """Convenience one-shot wrapper around KNNScorer (kept for smoke_reproduce.py)."""
    return KNNScorer(x_train, n_neighbors)(x_eval)


def benign_attack_indices(data):
    """Return (benign_idx, attack_idx) over the full dataset using its labels."""
    labels = np.array([data["dataset"][i][1] for i in data["full_index"]])
    benign = np.where(labels == 0)[0]
    attack = np.where(labels == 1)[0]
    return benign, attack
