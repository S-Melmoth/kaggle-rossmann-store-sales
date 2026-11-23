from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import catboost as cb

from src.data import HISTORY_COLUMNS


CATEGORICAL = ["Store", "DayOfWeek", "Month"]
EXPERIMENTS = {
    "no_history": [],
    "with_history": HISTORY_COLUMNS,
    "without_trend": [name for name in HISTORY_COLUMNS if name != "trend"],
    "without_last_year": [name for name in HISTORY_COLUMNS if name != "last_year"],
    "with_weekday_mean": ["weekday_mean" if name == "weekday_median" else name for name in HISTORY_COLUMNS],
}


def fit_model(train: pd.DataFrame, history_columns: List[str]):
    prepare = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numbers", StandardScaler(), ["Promo", *history_columns]),
        ],
        sparse_threshold=1.0,
    )
    model = TransformedTargetRegressor(
        regressor=make_pipeline(prepare, LinearRegression()),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    positive = train.loc[train["Sales"] > 0]
    with threadpool_limits(limits=4, user_api="blas"):
        return model.fit(positive, positive["Sales"])


def fit_linear_or_ridge(
    train: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    is_ridge: bool = False,
    alpha: float = 10.0,
    random_state: int = 42,
) -> TransformedTargetRegressor:
    num_cols = [c for c in feature_cols if c not in cat_cols]
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    preprocessor = ColumnTransformer(transformers, sparse_threshold=1.0)
    base_reg = Ridge(alpha=alpha, random_state=random_state) if is_ridge else LinearRegression()

    model = TransformedTargetRegressor(
        regressor=make_pipeline(preprocessor, base_reg),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    positive = train.loc[train["Sales"] > 0]
    with threadpool_limits(limits=4, user_api="blas"):
        model.fit(positive[feature_cols], positive["Sales"])
    return model


def fit_catboost(
    train: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.1,
    random_seed: int = 42,
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    verbose: int = 0,
) -> cb.CatBoostRegressor:
    train_pos = train.loc[train["Sales"] > 0].copy()
    for col in cat_cols:
        train_pos[col] = train_pos[col].astype(str)

    cb_eval_set = None
    if eval_set is not None:
        eval_x, eval_y = eval_set
        eval_x_copy = eval_x.copy()
        for col in cat_cols:
            eval_x_copy[col] = eval_x_copy[col].astype(str)
        cb_eval_set = (eval_x_copy[feature_cols], np.log1p(eval_y))

    model = cb.CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_cols,
        random_seed=random_seed,
        thread_count=-1,
        verbose=verbose,
    )
    model.fit(
        train_pos[feature_cols],
        np.log1p(train_pos["Sales"]),
        eval_set=cb_eval_set,
        verbose=verbose,
    )
    return model


def predict(
    model,
    rows: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    is_catboost: bool = False,
) -> np.ndarray:
    if is_catboost and feature_cols is not None:
        sub = rows[feature_cols].copy()
        if cat_cols:
            for col in cat_cols:
                sub[col] = sub[col].astype(str)
        raw_pred = model.predict(sub)
        pred = np.maximum(0, np.expm1(raw_pred))
    else:
        if feature_cols is not None:
            pred = np.maximum(0, model.predict(rows[feature_cols]))
        else:
            pred = np.maximum(0, model.predict(rows))

    # Store closed rule: if Open == 0, Sales is 0
    return np.where(rows["Open"] == 0, 0, pred)
