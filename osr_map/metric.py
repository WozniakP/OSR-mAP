import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


# Computes Average Precision (AP) for a single query
# based on a binary relevance vector.
# rel[k] = 1 means that the item at rank k is relevant.
def _compute_ap(rel):
    rel = np.asarray(rel).astype(np.int32)

    if rel.sum() == 0:
        return 0.0

    correct = 0
    precisions = []

    for i, r in enumerate(rel, start=1):
        if r:
            correct += 1
            precisions.append(correct / i)

    return float(np.mean(precisions))


# Computes Open-Set Retrieval mean Average Precision (OSR-mAP).
#
# This metric evaluates retrieval quality under open-set conditions.
# It combines:
# - AP for known queries,
# - rejection of unknown queries using distance thresholds,
# - area under the KAP-vs-FAR curve.
#
# ranking_labels defines the evaluated ranking depth K,
# so the score can be interpreted as OSR-mAP@K.
def osr_map_score(
    y_true,
    ranking_labels,
    distances,
    ref_set,
    return_curve=False,
    plot=False
):
    """
    Open-Set Retrieval mean Average Precision (OSR-mAP).

    Parameters
    ----------
    y_true : array-like of shape (n_queries,)
        Ground-truth query labels.

    ranking_labels : array-like of shape (n_queries, rank_len)
        Ranked retrieved labels for each query.
        The ranking length defines K, so this is effectively OSR-mAP@K.

    distances : array-like of shape (n_queries,)
        Distance / uncertainty score for each query.
        Lower value means more confident / more likely known.

    ref_set : set
        Set of known classes.

    return_curve : bool, default=False
        If True, returns score, FAR, KAP and mode.

    plot : bool, default=False
        If True, plots the FAR-KAP curve.

    Returns
    -------
    float or dict
        If return_curve=False, returns only OSR-mAP score.
        If return_curve=True, returns a dictionary.
    """

    y_true = np.asarray(y_true)
    ranking_labels = np.asarray(ranking_labels)
    distances = np.asarray(distances, dtype=np.float32)
    ref_set = set(ref_set)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")

    if ranking_labels.ndim != 2:
        raise ValueError("ranking_labels must be a 2D array: (n_queries, rank_len).")

    if distances.ndim != 1:
        raise ValueError("distances must be a 1D array.")

    if len(y_true) != len(ranking_labels):
        raise ValueError("y_true and ranking_labels must have the same number of queries.")

    if len(y_true) != len(distances):
        raise ValueError("y_true and distances must have the same number of queries.")

    known_mask = np.array([g in ref_set for g in y_true])
    unknown_mask = ~known_mask

    num_known = int(known_mask.sum())
    num_unknown = int(unknown_mask.sum())

    # -------------------------
    # AP per query
    # -------------------------
    ap_array = np.zeros(len(y_true), dtype=np.float32)

    for i, g in enumerate(y_true):
        if g not in ref_set:
            continue

        rel = (ranking_labels[i] == g).astype(np.int32)
        ap_array[i] = _compute_ap(rel)

    # -------------------------
    # Unknown-only case
    # -------------------------
    if num_known == 0:
        result = {
            "score": None,
            "mode": "unknown-only",
            "far": None,
            "kap": None
        }
        return result if return_curve else None

    # -------------------------
    # Closed-set case
    # -------------------------
    if num_unknown == 0:
        score = float(ap_array[known_mask].mean())

        result = {
            "score": score,
            "mode": "closed-set",
            "far": None,
            "kap": None
        }

        return result if return_curve else score

    # -------------------------
    # Open-set case
    # -------------------------
    eps = 1e-9

    thresholds = np.unique(distances)
    thresholds = np.concatenate([
        [distances.min() - eps],  # reject all
        thresholds,
        [distances.max() + eps]   # accept all
    ])

    far_list = []
    kap_list = []

    for t in thresholds:
        accepted = distances <= t

        far = np.sum(accepted & unknown_mask) / num_unknown

        kap = np.sum(
            ap_array[known_mask] * accepted[known_mask]
        ) / num_known

        far_list.append(far)
        kap_list.append(kap)

    far = np.asarray(far_list, dtype=np.float32)
    kap = np.asarray(kap_list, dtype=np.float32)

    # Sort by FAR
    order = np.argsort(far)
    far = far[order]
    kap = kap[order]

    # For duplicated FAR values, keep the best KAP
    unique_far = []
    best_kap = []

    for f in np.unique(far):
        mask = far == f
        unique_far.append(f)
        best_kap.append(np.max(kap[mask]))

    far = np.asarray(unique_far, dtype=np.float32)
    kap = np.asarray(best_kap, dtype=np.float32)

    score = float(auc(far, kap)) if len(far) > 1 else 0.0

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(far, kap, marker="o", label=f"OSR-mAP = {score:.4f}")
        plt.xlabel("FAR (False Acceptance Rate)")
        plt.ylabel("KAP (Known AP)")
        plt.title("OSR-mAP Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

    result = {
        "score": score,
        "mode": "open-set",
        "far": far,
        "kap": kap
    }

    return result if return_curve else score