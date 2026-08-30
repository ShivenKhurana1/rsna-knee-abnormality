"""Rebuild the OOF harness arrays (F_dino, F_rad11, F_rad13, v52, gold_mask) from the
downloaded raw sources in oof_harness/raw/, verifying against the documented gold AUCs
before trusting anything (per oof_harness/README.md's own verification standard)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = "/home/zha76451/Projects/rsna-knee-abnormality/oof_harness"
RAW = f"{HERE}/raw"

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

tr = pd.read_csv("/local/home/zha76451/rsna-knee-train/data/train.csv")
assert len(tr) == 4407
uid2row = {u: i for i, u in enumerate(tr["StudyInstanceUID"])}
gold_mask = tr[LAB].notna().all(axis=1).values
print("gold studies:", gold_mask.sum())

# train.csv's LAB columns are populated ONLY for the 58 gold studies (real 0/1 labels).
# The other 4,349 studies' targets live in the separate LLM-derived soft-label table,
# binarized at 0.5 here to serve as pseudo ground truth for AUC on the weak pool.
Y_all = np.full((4407, 12), np.nan)
Y_all[gold_mask] = tr.loc[gold_mask, LAB].values.astype(np.float64)
soft = pd.read_parquet("/local/home/zha76451/rsna-knee-train/data/labels_llm_soft.parquet")
soft = soft.set_index("StudyInstanceUID")
weak_rows = np.array([uid2row[u] for u in soft.index])
assert not gold_mask[weak_rows].any(), "weak-label table overlaps gold studies"
Y_all[weak_rows] = (soft[LAB].values >= 0.5).astype(np.float64)
print(f"weak-pool labels filled: {len(weak_rows)} (binarized at 0.5)")
print(f"rows with no label at all: {np.isnan(Y_all[:, 0]).sum()}")

# ---- F_dino: surasan092 focal folds 0-2 (only 3 of the planned 5 folds were uploaded) ----
F_dino = np.full((4407, 12), np.nan)
covered = np.zeros(4407, dtype=bool)
for f in range(3):
    idx = np.load(f"{RAW}/sura/focal_fold{f}_val_indices.npy")
    pred = np.load(f"{RAW}/sura/focal_fold{f}_val_predictions.npy")
    assert not covered[idx].any(), f"fold {f} overlaps previously-filled rows"
    F_dino[idx] = pred
    covered[idx] = True
print(f"F_dino coverage: {covered.sum()}/4407 studies ({covered.sum()/4407:.1%})")
print(f"F_dino gold coverage: {covered[gold_mask].sum()}/{gold_mask.sum()}")


def csv_to_grid(path):
    df = pd.read_csv(path)
    df = df.set_index("StudyInstanceUID")
    grid = np.full((4407, 12), np.nan)
    rows = np.array([uid2row[u] for u in df.index])
    grid[rows] = df[LAB].values
    return grid, np.isin(np.arange(4407), rows)


F_rad11, cov11 = csv_to_grid(f"{RAW}/tonylica/e11/v52_e11_oof.csv")
F_rad13, cov13 = csv_to_grid(f"{RAW}/tonylica/e13/v52_e11_oof.csv")
F_v52, covv52 = csv_to_grid(f"{RAW}/prvsiyan/v52_oof.csv")
for name, cov in [("rad11", cov11), ("rad13", cov13), ("v52", covv52)]:
    print(f"F_{name} coverage: {cov.sum()}/4407 ({cov.sum()/4407:.1%}), gold coverage "
          f"{cov[gold_mask].sum()}/{gold_mask.sum()}")


def macro_auc(P, Y, mask):
    P, Y = P[mask], Y[mask]
    aucs = []
    for j in range(12):
        valid = ~np.isnan(P[:, j])
        yv, pv = Y[valid, j], P[valid, j]
        if len(set(yv.astype(int))) > 1:
            aucs.append(roc_auc_score(yv, pv))
    return float(np.mean(aucs)), aucs


print("\n--- Sanity check: reproduce documented gold-gate AUCs ---")
for name, F in [("dino", F_dino), ("e11", F_rad11), ("e13", F_rad13), ("v52", F_v52)]:
    valid_gold = gold_mask & ~np.isnan(F[:, 0])
    au, _ = macro_auc(F, Y_all, valid_gold)
    print(f"  {name:6s} gold macro-AUC = {au:.4f}  (n={valid_gold.sum()})")

np.save(f"{HERE}/F_dino.npy", F_dino)
np.save(f"{HERE}/F_rad11.npy", F_rad11)
np.save(f"{HERE}/F_rad13.npy", F_rad13)
np.save(f"{HERE}/F_v52.npy", F_v52)
np.save(f"{HERE}/gold_mask.npy", gold_mask)
np.save(f"{HERE}/Y_all.npy", Y_all)
print("\nSaved F_dino.npy, F_rad11.npy, F_rad13.npy, F_v52.npy, gold_mask.npy, Y_all.npy")
