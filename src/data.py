from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


KEY = ["Store", "Date"]
INPUT_COLUMNS = KEY + ["DayOfWeek", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
HISTORY_WEEKS = 8
TREND_EMA_SPAN = 3

HISTORY_COLUMNS = [
    "store_median",
    "weekday_median",
    "recent_mean",
    "trend",
    "last_year",
    "promo_median",
    "sales_28_days_before_cutoff",
    "recent_std",
    "recent_weekday_median",
]

STORE_META_COLUMNS = [
    "StoreType",
    "Assortment",
    "log_CompetitionDistance",
    "has_competition",
    "is_competition_open",
    "competition_age_months",
    "Promo2",
    "is_promo2_active",
    "promo2_age_weeks",
]

PROMO_SPLIT_COLUMNS = [
    "recent_mean_promo",
    "recent_mean_no_promo",
    "store_median_promo",
    "store_median_no_promo",
    "promo_uplift",
    "expected_promo_sales",
]

CALENDAR_COLUMNS = [
    "Day",
    "WeekOfYear",
    "IsMonthStart",
    "IsMonthEnd",
    "IsMidMonth",
    "SchoolHoliday",
]

NOISY_FEATURE_COLUMNS = [
    "recent_7d_mean",
]

EXTENDED_COLUMNS = [
    "promo_weekday_median",
    "is_weekend",
    "days_to_month_end",
    "week_sin",
    "week_cos",
    "competition_near",
    "sales_variation_coef",
    "last_year_ratio",
    "sales_momentum_4w_8w",
    "store_type_assortment",
]

MONTH_ABBR_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"
}


def load_sales(path: Path) -> pd.DataFrame:
    sales = pd.read_csv(path, parse_dates=["Date"], dtype={"StateHoliday": str})
    if sales.empty or sales[INPUT_COLUMNS].isna().any().any() or sales.duplicated(KEY).any():
        raise ValueError("Sales data must have complete, unique store-date rows.")
    if (sales["Sales"] < 0).any():
        raise ValueError("Sales cannot be negative.")
    sales["Month"] = sales["Date"].dt.month
    sales["StateHoliday"] = sales["StateHoliday"].fillna("0").astype(str)
    return sales.sort_values("Date").reset_index(drop=True)


def load_store(path: Path) -> pd.DataFrame:
    store = pd.read_csv(path)
    required = ["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]
    if store.empty or any(col not in store.columns for col in required):
        raise ValueError("Store data must contain required metadata columns.")
    return store.sort_values("Store").reset_index(drop=True)


def smoothed_weekly_slope(weekly_sales: pd.Series) -> float:
    if len(weekly_sales) < 2:
        return 0.0
    smoothed = weekly_sales.ewm(span=TREND_EMA_SPAN, adjust=False).mean()
    return float(np.polyfit(weekly_sales.index, smoothed, 1)[0])


def add_store_metadata(rows: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(store, on="Store", how="left")

    # Competition features
    max_dist = store["CompetitionDistance"].max()
    dist = merged["CompetitionDistance"].fillna(max_dist * 2 if pd.notna(max_dist) else 100000.0)
    merged["log_CompetitionDistance"] = np.log1p(dist)
    merged["has_competition"] = merged["CompetitionDistance"].notna().astype(int)

    comp_year = merged["CompetitionOpenSinceYear"].fillna(9999).astype(int)
    comp_month = merged["CompetitionOpenSinceMonth"].fillna(12).astype(int)
    is_open = (
        (merged["Date"].dt.year > comp_year)
        | ((merged["Date"].dt.year == comp_year) & (merged["Date"].dt.month >= comp_month))
    )
    merged["is_competition_open"] = is_open.astype(int)
    comp_age = (merged["Date"].dt.year - comp_year) * 12 + (merged["Date"].dt.month - comp_month)
    merged["competition_age_months"] = np.where(is_open, np.maximum(0, comp_age), 0)

    # Promo2 features
    p2_year = merged["Promo2SinceYear"].fillna(9999).astype(int)
    p2_week = merged["Promo2SinceWeek"].fillna(99).astype(int)
    p2_started = (
        (merged["Date"].dt.year > p2_year)
        | ((merged["Date"].dt.year == p2_year) & (merged["Date"].dt.isocalendar().week >= p2_week))
    )
    cur_month_abbr = merged["Date"].dt.month.map(MONTH_ABBR_MAP)
    in_interval = np.array([
        (abbr in interval.split(",")) if isinstance(interval, str) else False
        for abbr, interval in zip(cur_month_abbr, merged["PromoInterval"])
    ], dtype=bool)

    merged["is_promo2_active"] = ((merged["Promo2"] == 1) & p2_started & in_interval).astype(int)
    p2_age = (merged["Date"].dt.year - p2_year) * 52 + (merged["Date"].dt.isocalendar().week - p2_week)
    merged["promo2_age_weeks"] = np.where(merged["Promo2"] == 1, np.maximum(0, p2_age), 0)

    # Keep only target rows columns and drop temporary raw store columns
    raw_store_cols = [
        "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"
    ]
    return merged.drop(columns=[c for c in raw_store_cols if c in merged.columns])


def add_history(
    rows: pd.DataFrame,
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    recent_days: int,
    year_lag_days: int,
    store_df: Optional[pd.DataFrame] = None,
    include_store_meta: bool = True,
    include_promo_split: bool = True,
    include_calendar: bool = True,
    include_noisy_7d: bool = False,
    include_extended_features: bool = False,
) -> pd.DataFrame:
    positive = history.loc[history["Sales"] > 0]
    store_sales = positive.groupby("Store")["Sales"]
    weekday_sales = positive.groupby(["Store", "DayOfWeek"])["Sales"]
    result = rows.copy()

    # 1. Baseline historical features
    result["store_median"] = rows["Store"].map(store_sales.median()).fillna(
        positive["Sales"].median()
    )
    weekday_keys = pd.MultiIndex.from_frame(rows[["Store", "DayOfWeek"]])
    result["weekday_median"] = weekday_sales.median().reindex(weekday_keys).to_numpy()
    result["weekday_median"] = result["weekday_median"].fillna(result["store_median"])
    result["weekday_mean"] = weekday_sales.mean().reindex(weekday_keys).to_numpy()
    result["weekday_mean"] = result["weekday_mean"].fillna(result["store_median"])

    recent_start = cutoff - pd.Timedelta(days=recent_days)
    recent = positive.loc[positive["Date"] >= recent_start].groupby("Store")["Sales"].mean()
    result["recent_mean"] = rows["Store"].map(recent).fillna(result["store_median"])

    eight_weeks_start = cutoff - pd.Timedelta(weeks=HISTORY_WEEKS)
    eight_weeks = positive.loc[positive["Date"] >= eight_weeks_start].copy()
    eight_weeks["week"] = (eight_weeks["Date"] - eight_weeks_start).dt.days // 7
    weekly_means = eight_weeks.groupby(["Store", "week"])["Sales"].mean()
    slopes = weekly_means.groupby(level="Store").apply(
        lambda sales: smoothed_weekly_slope(sales.droplevel("Store"))
    )
    result["trend"] = rows["Store"].map(slopes).fillna(0)

    cutoff_lag = cutoff - pd.Timedelta(days=28)
    cutoff_lag_sales = history.loc[history["Date"].eq(cutoff_lag)].set_index("Store")["Sales"]
    result["sales_28_days_before_cutoff"] = rows["Store"].map(cutoff_lag_sales).fillna(result["store_median"])
    recent_std = eight_weeks.groupby("Store")["Sales"].std()
    result["recent_std"] = rows["Store"].map(recent_std).fillna(0)
    recent_weekday = eight_weeks.groupby(["Store", "DayOfWeek"])["Sales"].median()
    result["recent_weekday_median"] = recent_weekday.reindex(weekday_keys).to_numpy()
    result["recent_weekday_median"] = result["recent_weekday_median"].fillna(result["weekday_median"])

    last_year_dates = rows["Date"] - pd.Timedelta(days=year_lag_days)
    last_year_keys = pd.MultiIndex.from_arrays([rows["Store"], last_year_dates], names=KEY)
    result["last_year"] = history.set_index(KEY)["Sales"].reindex(last_year_keys).to_numpy()
    result["last_year_available"] = result["last_year"].notna()
    result["last_year"] = result["last_year"].fillna(result["weekday_median"])

    promo_sales = positive.loc[positive["Promo"] == 1].groupby("Store")["Sales"].median()
    result["promo_median"] = rows["Store"].map(promo_sales).fillna(result["store_median"])
    result["promo_median"] = result["promo_median"] * rows["Promo"]

    # 2. Promo split features (addresses baseline overprediction on non-promo days)
    if include_promo_split:
        no_promo_sales = positive.loc[positive["Promo"] == 0].groupby("Store")["Sales"].median()
        result["store_median_promo"] = rows["Store"].map(promo_sales).fillna(result["store_median"])
        result["store_median_no_promo"] = rows["Store"].map(no_promo_sales).fillna(result["store_median"])

        recent_promo = positive.loc[(positive["Date"] >= recent_start) & (positive["Promo"] == 1)].groupby("Store")["Sales"].mean()
        recent_no_promo = positive.loc[(positive["Date"] >= recent_start) & (positive["Promo"] == 0)].groupby("Store")["Sales"].mean()
        result["recent_mean_promo"] = rows["Store"].map(recent_promo).fillna(result["store_median_promo"])
        result["recent_mean_no_promo"] = rows["Store"].map(recent_no_promo).fillna(result["store_median_no_promo"])

        result["promo_uplift"] = (result["store_median_promo"] - result["store_median_no_promo"]) / (result["store_median_no_promo"] + 1.0)
        result["expected_promo_sales"] = np.where(
            rows["Promo"] == 1,
            result["recent_mean_promo"],
            result["recent_mean_no_promo"]
        )

    # 3. Calendar features
    if include_calendar:
        result["Day"] = rows["Date"].dt.day
        result["WeekOfYear"] = rows["Date"].dt.isocalendar().week.astype(int)
        result["IsMonthStart"] = (rows["Date"].dt.day <= 5).astype(int)
        result["IsMonthEnd"] = (rows["Date"].dt.day >= 25).astype(int)
        result["IsMidMonth"] = rows["Date"].dt.day.between(10, 16).astype(int)
        result["SchoolHoliday"] = rows["SchoolHoliday"].astype(int)

    # 4. Noisy 7-day feature (for deliberate negative experiment)
    if include_noisy_7d:
        recent_7d_start = cutoff - pd.Timedelta(days=7)
        recent_7d = positive.loc[positive["Date"] >= recent_7d_start].groupby("Store")["Sales"].mean()
        result["recent_7d_mean"] = rows["Store"].map(recent_7d).fillna(result["store_median"])

    # 5. Store metadata & competition features
    if include_store_meta and store_df is not None:
        result = add_store_metadata(result, store_df)

    # 6. Extended features (saturation test)
    if include_extended_features:
        result["promo_weekday_median"] = (result["weekday_median"] * rows["Promo"]).astype(float)
        result["is_weekend"] = (rows["DayOfWeek"] >= 6).astype(int)
        result["days_to_month_end"] = (rows["Date"].dt.days_in_month - rows["Date"].dt.day).astype(int)
        week_float = rows["Date"].dt.isocalendar().week.astype(float)
        result["week_sin"] = np.sin(2.0 * np.pi * week_float / 52.1429)
        result["week_cos"] = np.cos(2.0 * np.pi * week_float / 52.1429)
        if "CompetitionDistance" in result.columns:
            result["competition_near"] = (result["CompetitionDistance"].fillna(99999) < 1000).astype(int)
        else:
            result["competition_near"] = 0
        result["sales_variation_coef"] = (result["recent_std"] / (result["store_median"] + 1.0)).astype(float)
        result["last_year_ratio"] = (result["last_year"] / (result["store_median"] + 1.0)).astype(float)
        result["sales_momentum_4w_8w"] = (result["recent_mean"] / (result["store_median"] + 1.0)).astype(float)
        if "StoreType" in result.columns and "Assortment" in result.columns:
            result["store_type_assortment"] = result["StoreType"].astype(str) + "_" + result["Assortment"].astype(str)
        else:
            result["store_type_assortment"] = "unknown"

    return result


def prepare_windows(
    sales: pd.DataFrame,
    config: dict,
    store_df: Optional[pd.DataFrame] = None,
    include_store_meta: bool = True,
    include_promo_split: bool = True,
    include_calendar: bool = True,
    include_noisy_7d: bool = False,
    include_extended_features: bool = False,
):
    window_days = config["window_days"]
    recent_days = config["recent_days"]
    year_lag_days = config["year_lag_days"]
    if not 0 < window_days <= year_lag_days or recent_days < 1:
        raise ValueError("Use a positive window no longer than the year lag and positive recent days.")

    window = pd.Timedelta(days=window_days)
    cutoff = sales["Date"].max() + pd.Timedelta(days=1) - window
    history_days = max(7 * HISTORY_WEEKS, recent_days)
    first_training_cutoff = sales["Date"].min() + pd.Timedelta(days=history_days)
    origins = pd.date_range(start=cutoff, end=first_training_cutoff, freq=-window)
    origins = origins[origins < cutoff].sort_values()
    if origins.empty:
        raise ValueError("Not enough history for a training window and a validation window.")

    def make_window(origin):
        past = sales.loc[sales["Date"] < origin]
        target = sales.loc[sales["Date"].between(origin, origin + window, inclusive="left")]
        if target.empty or not (past["Sales"] > 0).any():
            raise ValueError(f"Missing history or targets for the window starting {origin.date()}.")
        return add_history(
            target, past, origin, recent_days, year_lag_days,
            store_df=store_df,
            include_store_meta=include_store_meta,
            include_promo_split=include_promo_split,
            include_calendar=include_calendar,
            include_noisy_7d=include_noisy_7d,
            include_extended_features=include_extended_features,
        )

    train = pd.concat([make_window(origin) for origin in origins], ignore_index=True)
    validation = make_window(cutoff).reset_index(drop=True)
    return train, validation, cutoff, origins
