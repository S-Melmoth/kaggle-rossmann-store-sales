import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set consistent publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8.5
plt.rcParams["figure.titlesize"] = 12


def plot_experiments_comparison(summary_data: list, output_path: Path):
    exp_map = {x["id"]: x for x in summary_data}

    exp_order = ["exp5_catboost_extended", "exp4_catboost_full", "exp0_baseline_linear", "exp1_ridge", "exp2_noisy_7d_lag", "exp3_features_linear"][::-1]
    labels = [
        "CatBoost Ext (41 призн.)\n(RMSPE: 15.29% | MAE: 737.1)",
        "CatBoost Full (31 призн.)\n(RMSPE: 15.11% | MAE: 732.8)",
        "Baseline Linear\n(RMSPE: 18.83% | MAE: 869.5)",
        "Ridge (L2)\n(RMSPE: 18.92% | MAE: 870.8)",
        "7-day lag (шум)\n(RMSPE: 19.36% | MAE: 892.4)",
        "Линейная (31 призн.)\n(RMSPE: 20.60% | MAE: 1009.5)",
    ][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    y_pos = np.arange(len(exp_order))
    deltas = []
    ci_err_left = []
    ci_err_right = []
    colors = []

    for eid in exp_order:
        item = exp_map[eid]
        if eid == "exp0_baseline_linear":
            deltas.append(0.0)
            ci_err_left.append(0.0)
            ci_err_right.append(0.0)
            colors.append("black")
        else:
            b3w = item.get("bootstrap_vs_baseline_3w", {})
            rel = b3w.get("relative_improvement_point", 0) * 100
            ci = [v * 100 for v in b3w.get("relative_improvement_95_ci", [0, 0])]
            d = -rel
            low = -ci[1]
            high = -ci[0]
            c = "#2ca02c" if d < 0 else "#d62728"
            deltas.append(d)
            ci_err_left.append(d - low)
            ci_err_right.append(high - d)
            colors.append(c)

    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    for i in range(len(y_pos)):
        if exp_order[i] == "exp0_baseline_linear":
            ax1.plot(0, y_pos[i], "s", color="black", markersize=6)
            ax1.text(0.8, y_pos[i] + 0.22, "0.00% (база)", ha="left", va="bottom", fontsize=8.5, fontweight="bold", color="black")
        else:
            ax1.errorbar(
                deltas[i], y_pos[i],
                xerr=[[ci_err_left[i]], [ci_err_right[i]]],
                fmt="o", color=colors[i], ecolor=colors[i],
                elinewidth=2.5, capsize=4, markersize=6
            )
            sign = "+" if deltas[i] > 0 else ""
            ax1.text(deltas[i], y_pos[i] + 0.22, f"{sign}{deltas[i]:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=colors[i])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Изменение ошибки к Baseline (%) (меньше 0 - лучше)", fontsize=9)
    ax1.set_title("95% бутстрап-интервалы изменений (3 окна)", fontsize=10, pad=10)
    ax1.set_xlim(-26, 20)
    ax1.set_ylim(-0.6, len(y_pos) - 0.3)
    ax1.grid(True, linestyle="--", alpha=0.5)

    windows = ["Зима\n(03.01-13.02)", "Весна\n(28.03-08.05)", "Лето\n(20.06-31.07)"]
    x_w = np.arange(len(windows))
    w = 0.35

    base_scores = [v * 100 for v in exp_map["exp0_baseline_linear"]["multi_window"]["rmspe_scores"]]
    cb_scores = [v * 100 for v in exp_map["exp4_catboost_full"]["multi_window"]["rmspe_scores"]]
    base_maes = exp_map["exp0_baseline_linear"]["multi_window"]["mae_scores"]
    cb_maes = exp_map["exp4_catboost_full"]["multi_window"]["mae_scores"]

    r1 = ax2.bar(x_w - w/2, base_scores, w, label="Baseline Linear", color="#1f77b4", alpha=0.85, edgecolor="black")
    r2 = ax2.bar(x_w + w/2, cb_scores, w, label="CatBoost Full", color="#2ca02c", alpha=0.85, edgecolor="black")

    for rect, mae in zip(r1, base_maes):
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%\n(MAE {mae:.0f})", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect, mae in zip(r2, cb_maes):
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%\n(MAE {mae:.0f})", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x_w)
    ax2.set_xticklabels(windows, fontsize=8.5)
    ax2.set_ylabel("RMSPE (%) (меньше - лучше)", fontsize=9)
    ax2.set_title("RMSPE и MAE по сезонам: Baseline vs CatBoost", fontsize=10, pad=10)
    ax2.set_ylim(0, max(base_scores) * 1.35)
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_residuals_distribution(base_preds: pd.DataFrame, cb_preds: pd.DataFrame, output_path: Path):
    scored_base = base_preds.loc[base_preds["Sales"] > 0]
    scored_cb = cb_preds.loc[cb_preds["Sales"] > 0]

    base_err = (scored_base["Prediction"] - scored_base["Sales"]) / scored_base["Sales"]
    cb_err = (scored_cb["Prediction"] - scored_cb["Sales"]) / scored_cb["Sales"]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    bins = np.linspace(-0.6, 0.8, 80)
    ax.hist(base_err, bins=bins, alpha=0.5, label=f"Baseline Linear (Std={base_err.std():.3f}, RMSPE=15.6%)", color="#1f77b4", density=True)
    ax.hist(cb_err, bins=bins, alpha=0.6, label=f"CatBoost Full (Std={cb_err.std():.3f}, RMSPE=12.2%)", color="#2ca02c", density=True)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Относительная ошибка: (Prediction - Fact) / Fact")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title("Распределение относительных ошибок: Baseline vs CatBoost Full", pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(-0.6, 0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_slice_analysis(base_metrics: dict, cb_metrics: dict, output_path: Path):
    categories = ["Общий RMSPE", "Низкие продажи (≤10%)", "Дни без акции (Promo=0)", "Дни с акцией (Promo=1)"]

    base_vals = [
        base_metrics["main_window"]["rmspe"] * 100,
        base_metrics["slices"]["low_sales"]["rmspe"] * 100,
        base_metrics["slices"]["no_promo"]["rmspe"] * 100,
        base_metrics["slices"]["promo"]["rmspe"] * 100,
    ]

    cb_vals = [
        cb_metrics["main_window"]["rmspe"] * 100,
        cb_metrics["slices"]["low_sales"]["rmspe"] * 100,
        cb_metrics["slices"]["no_promo"]["rmspe"] * 100,
        cb_metrics["slices"]["promo"]["rmspe"] * 100,
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline Linear", color="#1f77b4", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, cb_vals, width, label="CatBoost + Promo Split", color="#2ca02c", alpha=0.85, edgecolor="black")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("RMSPE (%) (меньше - лучше)")
    ax.set_title("Анализ ошибок по ключевым срезам: устранение слабостей Baseline", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.legend()
    ax.set_ylim(0, max(base_vals) * 1.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_feature_importance(fi_dict: dict, output_path: Path):
    if not fi_dict:
        return
    sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:18]
    features = [x[0] for x in sorted_fi][::-1]
    scores = [x[1] for x in sorted_fi][::-1]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.barh(features, scores, color="#2ca02c", alpha=0.8, edgecolor="black", height=0.65)
    ax.set_xlabel("Важность признака (Feature Importance)")
    ax.set_title("Топ-18 наиболее важных признаков в CatBoost", pad=10)

    for bar, val in zip(bars, scores):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", ha="left", fontsize=8)

    ax.set_xlim(0, max(scores) * 1.15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_store_timelines(base_df: pd.DataFrame, cb_df: pd.DataFrame, output_path: Path):
    stores_to_plot = [5, 551, 475]
    store_titles = {
        5: "Магазин 5 (Слабое место Baseline: малые чеки в субботу)",
        551: "Магазин 551 (Четверг с акцией Promo)",
        475: "Магазин 475 (Понедельник без акции после промо)",
    }

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.0), sharex=True)

    for ax, store_id in zip(axes, stores_to_plot):
        sub_base = base_df.loc[base_df["Store"] == store_id].sort_values("Date")
        sub_cb = cb_df.loc[cb_df["Store"] == store_id].sort_values("Date")

        ax.plot(sub_base["Date"], sub_base["Sales"], label="Факт (Actual)", color="black", linewidth=1.5, marker="o", markersize=3)
        ax.plot(sub_base["Date"], sub_base["Prediction"], label="Baseline Linear", color="#1f77b4", linestyle="--", linewidth=1.4)
        ax.plot(sub_cb["Date"], sub_cb["Prediction"], label="CatBoost Full", color="#2ca02c", linewidth=1.5)

        ax.set_title(store_titles.get(store_id, f"Магазин {store_id}"), fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Выручка", fontsize=8.5)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Дата (2015-06-20 по 2015-07-31)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_learning_curve(output_path: Path):
    from src.data import load_sales, load_store, prepare_windows, HISTORY_COLUMNS, STORE_META_COLUMNS, PROMO_SPLIT_COLUMNS, CALENDAR_COLUMNS
    import catboost as cb

    sales = load_sales(Path("data/train.csv"))
    store = load_store(Path("data/store.csv"))
    config = json.loads(Path("configs/hw2.json").read_text(encoding="utf-8"))

    train, val, _, _ = prepare_windows(sales, config, store_df=store)
    base_cols = ["Store", "DayOfWeek", "Month", "Promo", "StateHoliday"]
    features = list(dict.fromkeys(base_cols + HISTORY_COLUMNS + STORE_META_COLUMNS + PROMO_SPLIT_COLUMNS + CALENDAR_COLUMNS))
    cat_cols = [c for c in ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"] if c in features]

    train_pos = train[train["Sales"] > 0].copy()
    val_pos = val[val["Sales"] > 0].copy()
    for col in cat_cols:
        train_pos[col] = train_pos[col].astype(str)
        val_pos[col] = val_pos[col].astype(str)

    model = cb.CatBoostRegressor(
        iterations=350, depth=6, learning_rate=0.12, loss_function="RMSE", eval_metric="RMSE",
        cat_features=cat_cols, random_seed=42, verbose=0, thread_count=-1
    )
    model.fit(train_pos[features], np.log1p(train_pos["Sales"]), eval_set=(val_pos[features], np.log1p(val_pos["Sales"])), verbose=0)
    evals = model.get_evals_result()

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(evals["learn"]["RMSE"], label="Train RMSE (log scale)", color="#1f77b4", linewidth=1.8)
    ax.plot(evals["validation"]["RMSE"], label="Validation RMSE (log scale)", color="#2ca02c", linewidth=1.8)
    ax.set_xlabel("Итерация (деревья решений)")
    ax.set_ylabel("RMSE на log1p(Sales)")
    ax.set_title("Кривая обучения (Learning Curve) модели CatBoost Full", pad=10)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_plots(summary_json: Path, artifacts_dir: Path):
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_experiments_comparison(summary, plots_dir / "experiments_comparison.png")

    exp_map = {x["id"]: x for x in summary}
    base_res = exp_map.get("exp0_baseline_linear")
    cb_res = exp_map.get("exp4_catboost_full")

    if base_res and cb_res:
        plot_slice_analysis(base_res, cb_res, plots_dir / "slice_analysis_low_sales_promo.png")

        base_preds = pd.read_csv(artifacts_dir / "experiments/exp0_baseline_linear/predictions.csv", parse_dates=["Date"])
        cb_preds = pd.read_csv(artifacts_dir / "experiments/exp4_catboost_full/predictions.csv", parse_dates=["Date"])

        plot_residuals_distribution(base_preds, cb_preds, plots_dir / "residuals_distribution.png")
        plot_store_timelines(base_preds, cb_preds, plots_dir / "store_forecast_examples.png")

    if cb_res and cb_res.get("feature_importance"):
        plot_feature_importance(cb_res["feature_importance"], plots_dir / "feature_importance.png")

    plot_learning_curve(plots_dir / "learning_curve.png")

    print(f"All plots successfully generated in {plots_dir}")


if __name__ == "__main__":
    generate_all_plots(
        Path("artifacts/experiments_summary.json"),
        Path("artifacts"),
    )
