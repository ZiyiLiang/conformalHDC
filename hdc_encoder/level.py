"""Level hypervector encoding of tabular features (UCI HAR, ISOLET).

Every feature is quantized to one of `levels` levels; a sample HV bundles, over its
features, the feature's random item HV bound to its level's correlated HV.
"""
import numpy as np
import torch
import torch.nn.functional as F


def bipolar_sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def quantize_to_levels(X, levels):
    """Min-max scale every feature and round it to an integer level in [0, levels - 1]."""
    Xmin = X.min(axis=0)
    rng = np.maximum(X.max(axis=0) - Xmin, 1e-8)
    return np.clip(np.round((X - Xmin) / rng * (levels - 1)), 0, levels - 1).astype(np.int64)


def _rand_bip(shape, device):
    r = torch.randint(0, 2, shape, device=device, dtype=torch.int8)
    return r.float().mul_(2).sub_(1)


def make_im_cim(n_features, levels, dim, device):
    """Random item HVs (iM) and correlated level HVs (CiM).

    Level l flips the first l * dim // (levels - 1) coordinates of a random
    permutation of one base HV, so neighbouring levels are similar.
    """
    iM = _rand_bip((n_features, dim), device)
    base = _rand_bip((dim,), device)
    perm = torch.randperm(dim, device=device)
    position = torch.empty_like(perm)
    position[perm] = torch.arange(dim, device=device)
    n_flips = torch.arange(levels, device=device) * dim // max(1, levels - 1)
    return iM, torch.where(position < n_flips[:, None], -base, base)


@torch.no_grad()
def encode_levels(L, iM, CiM, batch_size=512):
    """Bipolar HVs sign(sum_f iM[f] * CiM[L[:, f]]) of a (n, n_features) level matrix."""
    n_features, n_levels = len(iM), len(CiM)
    # Row f * n_levels + l holds iM[f] * CiM[l]; each sample sums one row per feature.
    table = (iM[:, None, :] * CiM[None, :, :]).reshape(n_features * n_levels, -1)
    offsets = torch.arange(n_features, device=iM.device) * n_levels
    return np.concatenate([
        bipolar_sign(F.embedding_bag(
            torch.as_tensor(L[i:i + batch_size], device=iM.device) + offsets, table, mode="sum",
        )).cpu().numpy()
        for i in range(0, len(L), batch_size)
    ])
