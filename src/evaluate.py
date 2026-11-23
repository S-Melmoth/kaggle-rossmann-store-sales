from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def metrics(rows: pd.DataFrame, prediction: np.ndarray) -> Dict[str, Any]:
    scored = rows["Sales"] > 0
    if not scored.any():
        raise ValueError("RMSPE needs at least one row with positive sales.")
    actual = rows.loc[scored, "Sales"].to_numpy()
    pred = prediction[scored]
    error = pred - actual
    relative_error = error / actual
    return {
        "rows": int(len(actual)),
        "rmspe": float(np.sqrt(np.mean(relative_error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(relative_error)),
    }


def slice_metrics(rows: pd.DataFrame, prediction: np.ndarray) -> Dict[str, Any]:
    scored = rows["Sales"] > 0
    df = rows.loc[scored].copy()
    pred = prediction[scored]
    actual = df["Sales"].to_numpy()

    q10 = np.quantile(actual, 0.10)
    low_mask = actual <= q10
    low_err = (pred[low_mask] - actual[low_mask]) / actual[low_mask]

    promo_mask = df["Promo"].to_numpy() == 1
    no_promo_mask = ~promo_mask

    slices = {
        "overall": metrics(rows, prediction),
        "low_sales": {
            "threshold": float(q10),
            "rows": int(np.sum(low_mask)),
            "rmspe": float(np.sqrt(np.mean(low_err ** 2))),
            "mae": float(np.mean(np.abs(pred[low_mask] - actual[low_mask]))),
            "bias": float(np.mean(low_err)),
        },
        "promo": {
            "rows": int(np.sum(promo_mask)),
            "rmspe": float(np.sqrt(np.mean(((pred[promo_mask] - actual[promo_mask]) / actual[promo_mask]) ** 2))),
            "mae": float(np.mean(np.abs(pred[promo_mask] - actual[promo_mask]))),
            "bias": float(np.mean((pred[promo_mask] - actual[promo_mask]) / actual[promo_mask])),
        },
        "no_promo": {
            "rows": int(np.sum(no_promo_mask)),
            "rmspe": float(np.sqrt(np.mean(((pred[no_promo_mask] - actual[no_promo_mask]) / actual[no_promo_mask]) ** 2))),
            "mae": float(np.mean(np.abs(pred[no_promo_mask] - actual[no_promo_mask]))),
            "bias": float(np.mean((pred[no_promo_mask] - actual[no_promo_mask]) / actual[no_promo_mask])),
        },
    }

    if "StoreType" in df.columns:
        store_types = {}
        for st in sorted(df["StoreType"].unique()):
            st_mask = df["StoreType"].to_numpy() == st
            if st_mask.any():
                st_act = actual[st_mask]
                st_pred = pred[st_mask]
                store_types[str(st)] = {
                    "rows": int(np.sum(st_mask)),
                    "rmspe": float(np.sqrt(np.mean(((st_pred - st_act) / st_act) ** 2))),
                    "mae": float(np.mean(np.abs(st_pred - st_act))),
                }
        slices["store_type"] = store_types

    return slices


def relative_improvement(reference: float, candidate: float) -> Optional[float]:
    return (reference - candidate) / reference if reference else None


def summarize_data(sales: pd.DataFrame) -> Dict[str, Any]:
    gap_start, gap_end = "2014-07-01", "2014-12-31"
    observed = sales.loc[sales["Date"].between(gap_start, gap_end), "Store"].unique()
    missing_stores = sales["Store"].nunique() - len(observed)
    return {
        "rows": len(sales),
        "stores": int(sales["Store"].nunique()),
        "first_date": str(sales["Date"].min().date()),
        "last_date": str(sales["Date"].max().date()),
        "missing": sales.isna().sum().to_dict(),
        "types": sales.dtypes.astype(str).to_dict(),
        "zero_sales": int(sales["Sales"].eq(0).sum()),
        "closed_days": int(sales["Open"].eq(0).sum()),
        "open_zero_sales": int((sales["Open"].eq(1) & sales["Sales"].eq(0)).sum()),
        "promo_days": int(sales["Promo"].eq(1).sum()),
        "observation_gap": {
            "start": gap_start,
            "end": gap_end,
            "days": len(pd.date_range(gap_start, gap_end)),
            "stores": int(missing_stores),
        },
        "positive_sales_quantiles": sales.loc[sales["Sales"] > 0, "Sales"]
        .quantile([0, 0.01, 0.5, 0.99, 1]).to_dict(),
    }


def bootstrap_rmspe_by_store(
    rows: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    scored = rows["Sales"] > 0
    actual = rows.loc[scored, "Sales"].to_numpy()
    errors = pd.DataFrame({
        "store": rows.loc[scored, "Store"].to_numpy(),
        "reference": ((reference[scored] - actual) / actual) ** 2,
        "candidate": ((candidate[scored] - actual) / actual) ** 2,
        "rows": 1,
    })
    groups = errors.groupby("store")[["reference", "candidate", "rows"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    rmspe_samples = np.empty((samples, 2))
    for index in range(samples):
        drawn = rng.integers(len(groups), size=len(groups))
        reference_error, candidate_error, count = groups[drawn].sum(axis=0)
        rmspe_samples[index] = np.sqrt([reference_error / count, candidate_error / count])
    return rmspe_samples


def historical_spikes(past: pd.DataFrame) -> pd.DataFrame:
    positive = past.loc[past["Sales"] > 0].copy()
    weekday_median = positive.groupby(["Store", "DayOfWeek"])["Sales"].transform("median")
    positive["spike"] = positive["Sales"] >= 3 * weekday_median
    spike_rate = positive.groupby("Store")["spike"].mean()
    stores = spike_rate.index[(spike_rate > 0) & (spike_rate <= 0.01)]
    return positive.loc[positive["spike"] & positive["Store"].isin(stores)]


def bootstrap_by_store(rows, reference, candidate, samples, seed):
    rmspe_samples = bootstrap_rmspe_by_store(rows, reference, candidate, samples, seed)
    reference_rmspe, candidate_rmspe = rmspe_samples[rmspe_samples[:, 0] > 0].T
    valid = (reference_rmspe - candidate_rmspe) / reference_rmspe
    return np.quantile(valid, [0.025, 0.975]).tolist() if valid.size else None


def compare_rmspe_by_store(rows, predictions, comparisons, samples, seed):
    family_size = len(comparisons)
    tail = 0.05 / (2 * family_size)
    results = {}
    for name, mask, reference, candidate in comparisons:
        subset = rows.loc[mask]
        scored = subset["Sales"].gt(0)
        stores = subset.loc[scored, "Store"].nunique()
        if stores < 2:
            continue
        before = predictions[reference][mask]
        after = predictions[candidate][mask]
        estimates = bootstrap_rmspe_by_store(subset, before, after, samples, seed)
        differences = estimates[:, 1] - estimates[:, 0]
        low, high = np.quantile(differences, [tail, 1 - tail])
        results[name] = {
            "stores": int(stores),
            "rows": int(scored.sum()),
            "delta_rmspe": metrics(subset, after)["rmspe"] - metrics(subset, before)["rmspe"],
            "interval": [float(low), float(high)],
            "excludes_zero": bool(high < 0 or low > 0),
        }
    return {"samples": samples, "family_size": family_size, "comparisons": results}


def bootstrap_delta_rmspe(
    rows: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    rmspe_samples = bootstrap_rmspe_by_store(rows, reference, candidate, samples, seed)
    ref_samples = rmspe_samples[:, 0]
    cand_samples = rmspe_samples[:, 1]
    differences = cand_samples - ref_samples  # negative means candidate is better (lower RMSPE)
    low, high = np.quantile(differences, [0.025, 0.975])

    valid_rel = (ref_samples - cand_samples) / ref_samples
    rel_low, rel_high = np.quantile(valid_rel, [0.025, 0.975])

    ref_point = metrics(rows, reference)["rmspe"]
    cand_point = metrics(rows, candidate)["rmspe"]

    return {
        "delta_rmspe_point": float(cand_point - ref_point),
        "delta_rmspe_95_ci": [float(low), float(high)],
        "relative_improvement_point": float((ref_point - cand_point) / ref_point),
        "relative_improvement_95_ci": [float(rel_low), float(rel_high)],
        "excludes_zero": bool(high < 0 or low > 0),
    }
