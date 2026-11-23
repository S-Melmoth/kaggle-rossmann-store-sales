import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.data import (
    load_sales,
    load_store,
    prepare_windows,
    HISTORY_COLUMNS,
    STORE_META_COLUMNS,
    PROMO_SPLIT_COLUMNS,
    CALENDAR_COLUMNS,
    NOISY_FEATURE_COLUMNS,
    EXTENDED_COLUMNS,
)
from src.evaluate import metrics, slice_metrics, bootstrap_delta_rmspe
from src.train import fit_linear_or_ridge, fit_catboost, predict


EXPERIMENT_DEFINITIONS = [
    {
        "id": "exp0_baseline_linear",
        "name": "Baseline (Linear Regression)",
        "model_type": "linear",
        "is_ridge": False,
        "alpha": None,
        "iterations": None,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month"],
        "hypothesis": "Оригинальный baseline из ДЗ 1: линейная модель на log1p(Sales) с OHE магазинов и 9 историческими признаками.",
    },
    {
        "id": "exp1_ridge",
        "name": "Linear + Ridge L2 (alpha=10.0)",
        "model_type": "ridge",
        "is_ridge": True,
        "alpha": 10.0,
        "iterations": None,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month"],
        "hypothesis": "L2-регуляризация (Ridge) защитит линейную модель от переобучения, ограничит рост весов и улучшит обобщающую способность.",
    },
    {
        "id": "exp2_noisy_7d_lag",
        "name": "Ridge + 7d Sales Lag",
        "model_type": "ridge",
        "is_ridge": True,
        "alpha": 10.0,
        "iterations": None,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + NOISY_FEATURE_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month"],
        "hypothesis": "Средняя выручка магазина за 7 дней до окна прогноза поможет учесть свежий темп продаж и снизит ошибку.",
    },
    {
        "id": "exp3_features_linear",
        "name": "Linear (Ridge) + All Features",
        "model_type": "ridge",
        "is_ridge": True,
        "alpha": 10.0,
        "iterations": None,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + STORE_META_COLUMNS + PROMO_SPLIT_COLUMNS + CALENDAR_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"],
        "hypothesis": "Обогащение метаданными магазинов, конкуренцией, раздельными промо-медианами и календарем улучшит качество даже линейной модели.",
    },
    {
        "id": "exp4_catboost_full",
        "name": "CatBoost + All Features (Proposed Best)",
        "model_type": "catboost",
        "is_ridge": False,
        "alpha": None,
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.12,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + STORE_META_COLUMNS + PROMO_SPLIT_COLUMNS + CALENDAR_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"],
        "hypothesis": "Градиентный бустинг деревьев решений на полном наборе признаков выявит нелинейные взаимодействия (StoreType * Promo, расстояние до конкурентов, сезонность) и покажет максимальное снижение RMSPE.",
    },
    {
        "id": "exp5_catboost_extended",
        "name": "CatBoost Extended (41 признак)",
        "model_type": "catboost",
        "is_ridge": False,
        "alpha": None,
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.12,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + STORE_META_COLUMNS + PROMO_SPLIT_COLUMNS + CALENDAR_COLUMNS + EXTENDED_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday", "store_type_assortment"],
        "hypothesis": "Добавление 10 дополнительных инженерных признаков (взаимодействие Promo и дня недели, комбинация StoreType и Assortment, дни до конца месяца, коэффициент вариации, флаг близкого конкурента, темп 4w/8w, циклические косинусы недели) для проверки гипотезы о насыщении признакового пространства.",
    },
    {
        "id": "ablation1_no_store_meta",
        "name": "CatBoost Ablation 1 (No Store Meta)",
        "model_type": "catboost",
        "is_ridge": False,
        "alpha": None,
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.12,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + PROMO_SPLIT_COLUMNS + CALENDAR_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month", "StateHoliday"],
        "hypothesis": "Абляция 1: исключение метаданных магазинов (`store.csv`) изолированно показывает вклад информации о типе магазина, ассортименте и конкурентах.",
    },
    {
        "id": "ablation2_no_promo_split",
        "name": "CatBoost Ablation 2 (No Promo Split)",
        "model_type": "catboost",
        "is_ridge": False,
        "alpha": None,
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.12,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + STORE_META_COLUMNS + CALENDAR_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"],
        "hypothesis": "Абляция 2: исключение раздельных промо/не-промо медиан изолированно показывает вклад решения проблемы завышения прогноза в дни без акций.",
    },
    {
        "id": "ablation3_no_calendar",
        "name": "CatBoost Ablation 3 (No Calendar Features)",
        "model_type": "catboost",
        "is_ridge": False,
        "alpha": None,
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.12,
        "feature_cols": ["Promo"] + HISTORY_COLUMNS + STORE_META_COLUMNS + PROMO_SPLIT_COLUMNS,
        "cat_cols": ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"],
        "hypothesis": "Абляция 3: исключение календарных признаков (день месяца, неделя года, зарплатные пики) показывает вклад внутримесячной динамики.",
    },
]


def run_experiment_on_datasets(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    exp_def: dict,
    seed: int,
    is_main_eval: bool = False,
):
    feature_cols = exp_def["feature_cols"]
    cat_cols = [c for c in exp_def["cat_cols"] if c in feature_cols or c in ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"]]
    full_feature_list = list(dict.fromkeys(cat_cols + [c for c in feature_cols if c not in cat_cols]))

    model_type = exp_def["model_type"]
    t0 = time.time()
    if model_type == "linear":
        model = fit_linear_or_ridge(train_data, full_feature_list, cat_cols, is_ridge=False)
        pred = predict(model, val_data, full_feature_list, cat_cols, is_catboost=False)
    elif model_type == "ridge":
        model = fit_linear_or_ridge(train_data, full_feature_list, cat_cols, is_ridge=True, alpha=exp_def.get("alpha", 10.0), random_state=seed)
        pred = predict(model, val_data, full_feature_list, cat_cols, is_catboost=False)
    elif model_type == "catboost":
        model = fit_catboost(
            train_data, full_feature_list, cat_cols,
            iterations=exp_def.get("iterations", 350),
            depth=exp_def.get("depth", 6),
            learning_rate=exp_def.get("learning_rate", 0.12),
            random_seed=seed,
            verbose=0,
        )
        pred = predict(model, val_data, full_feature_list, cat_cols, is_catboost=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    fit_time = time.time() - t0
    m = metrics(val_data, pred)
    slices = slice_metrics(val_data, pred) if is_main_eval else None

    feature_importance = None
    if model_type == "catboost" and is_main_eval:
        fi = model.get_feature_importance()
        feature_importance = {name: float(val) for name, val in zip(full_feature_list, fi)}

    return {
        "metrics": m,
        "slices": slices,
        "predictions": pred,
        "fit_time": fit_time,
        "feature_importance": feature_importance,
    }


def run_all(config_path: Path, only_exp: Optional[str] = None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = config.get("seed", 42)
    sales = load_sales(Path(config["data_path"]))
    store = load_store(Path("data/store.csv" if Path("data/store.csv").exists() else "kaggle_rossmann_store_sales/data/store.csv"))

    window_delta = pd.Timedelta(days=config["window_days"])
    cutoff = sales["Date"].max() + pd.Timedelta(days=1) - window_delta

    eval_windows = [
        ("window_1", cutoff - 4 * window_delta, "2015-01-03", "2015-02-13"),
        ("window_2", cutoff - 2 * window_delta, "2015-03-28", "2015-05-08"),
        ("window_3_main", cutoff, "2015-06-20", "2015-07-31"),
    ]

    base_output = Path(config.get("output_dir", "artifacts/experiments"))
    base_output.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    configs_exp_dir = Path("configs/experiments")
    configs_exp_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path("artifacts/experiments_summary.json")
    existing_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    existing_map = {x["id"]: x for x in existing_summary}

    print("=== Step 1: Pre-computing feature datasets for all 3 validation windows ===", flush=True)
    window_datasets = {}
    for win_name, win_cutoff, win_start, win_end in eval_windows:
        t0 = time.time()
        print(f"Preparing window '{win_name}' (Cutoff: {win_cutoff.date()})...", flush=True)
        w_train, w_val, _, _ = prepare_windows(
            sales.loc[sales["Date"] < (win_cutoff + window_delta)],
            config,
            store_df=store,
            include_store_meta=True,
            include_promo_split=True,
            include_calendar=True,
            include_noisy_7d=True,
            include_extended_features=True,
        )
        print(f"  -> Done in {time.time() - t0:.1f}s (Train: {len(w_train):,}, Val: {len(w_val):,})", flush=True)
        window_datasets[win_name] = (w_train, w_val)

    main_val_df = window_datasets["window_3_main"][1]
    summary_results = []
    baseline_window_preds = {}
    all_window_preds = {}
    catboost_full_metrics = None

    print(f"\n=== Step 2: Running {len(EXPERIMENT_DEFINITIONS)} Experiments across all windows ===", flush=True)

    for exp_idx, exp_def in enumerate(EXPERIMENT_DEFINITIONS):
        exp_id = exp_def["id"]
        exp_name = exp_def["name"]

        exp_config_file = configs_exp_dir / f"{exp_id}.json"
        exp_config_file.write_text(json.dumps(exp_def, indent=2, ensure_ascii=False), encoding="utf-8")

        # When only_exp is requested, skip experiments that are not required for reference
        if only_exp and exp_id not in ["exp0_baseline_linear", "exp4_catboost_full", only_exp]:
            if exp_id in existing_map:
                print(f"[{exp_idx + 1}/{len(EXPERIMENT_DEFINITIONS)}] Skipping {exp_id} (using cached results)", flush=True)
                summary_results.append(existing_map[exp_id])
                continue

        print(f"\n[{exp_idx + 1}/{len(EXPERIMENT_DEFINITIONS)}] Running: {exp_name} ({exp_id})...", flush=True)

        exp_config_file = configs_exp_dir / f"{exp_id}.json"
        exp_config_file.write_text(json.dumps(exp_def, indent=2, ensure_ascii=False), encoding="utf-8")

        # 1. Main Window Evaluation
        main_train, main_val = window_datasets["window_3_main"]
        res_main = run_experiment_on_datasets(
            main_train, main_val, exp_def, seed, is_main_eval=True
        )

        main_metrics = res_main["metrics"]
        main_slices = res_main["slices"]
        main_preds = res_main["predictions"]

        # 2. Multi-window Evaluation
        multi_window_rmspe = []
        multi_window_mae = []
        current_window_preds = {}

        for win_name, _, _, _ in eval_windows:
            if win_name == "window_3_main":
                multi_window_rmspe.append(main_metrics["rmspe"])
                multi_window_mae.append(main_metrics["mae"])
                current_window_preds[win_name] = main_preds
            else:
                w_train, w_val = window_datasets[win_name]
                w_res = run_experiment_on_datasets(w_train, w_val, exp_def, seed, is_main_eval=False)
                multi_window_rmspe.append(w_res["metrics"]["rmspe"])
                multi_window_mae.append(w_res["metrics"]["mae"])
                current_window_preds[win_name] = w_res["predictions"]

        all_window_preds[exp_id] = current_window_preds

        mean_rmspe = float(np.mean(multi_window_rmspe))
        std_rmspe = float(np.std(multi_window_rmspe, ddof=1))
        mean_mae = float(np.mean(multi_window_mae))
        std_mae = float(np.std(multi_window_mae, ddof=1))

        if exp_id == "exp0_baseline_linear":
            baseline_window_preds = current_window_preds

        if exp_id == "exp4_catboost_full":
            catboost_full_metrics = {
                "w3_rmspe": main_metrics["rmspe"],
                "w3_mae": main_metrics["mae"],
                "mean_rmspe": mean_rmspe,
                "mean_mae": mean_mae,
            }

        # 3. Bootstrap CI vs Baseline (Window 3 and Pooled 3-Windows)
        bootstrap_info = None
        bootstrap_3w_info = None
        if baseline_window_preds:
            # Main window bootstrap
            bootstrap_info = bootstrap_delta_rmspe(
                main_val_df, baseline_window_preds["window_3_main"], main_preds,
                samples=config.get("bootstrap_samples", 1000),
                seed=seed,
            )
            # Pooled 3-window cluster bootstrap
            pooled_val = pd.concat([window_datasets[w[0]][1] for w in eval_windows], ignore_index=True)
            pooled_base = np.concatenate([baseline_window_preds[w[0]] for w in eval_windows])
            pooled_cand = np.concatenate([current_window_preds[w[0]] for w in eval_windows])
            bootstrap_3w_info = bootstrap_delta_rmspe(
                pooled_val, pooled_base, pooled_cand,
                samples=config.get("bootstrap_samples", 1000),
                seed=seed,
            )

        exp_dir = base_output / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_df = main_val_df[["Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "Sales"]].copy()
        pred_df["Prediction"] = main_preds
        pred_df.to_csv(exp_dir / "predictions.csv", index=False)

        exp_result = {
            "id": exp_id,
            "name": exp_name,
            "hypothesis": exp_def["hypothesis"],
            "model_type": exp_def["model_type"],
            "feature_count": len(exp_def["feature_cols"]),
            "fit_time_seconds": round(res_main["fit_time"], 2),
            "main_window": {
                "start": "2015-06-20",
                "end": "2015-07-31",
                "rmspe": main_metrics["rmspe"],
                "mae": main_metrics["mae"],
                "bias": main_metrics["bias"],
            },
            "multi_window": {
                "rmspe_scores": multi_window_rmspe,
                "mean_rmspe": mean_rmspe,
                "std_rmspe": std_rmspe,
                "mae_scores": multi_window_mae,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
            },
            "bootstrap_vs_baseline": bootstrap_info,
            "bootstrap_vs_baseline_3w": bootstrap_3w_info,
            "slices": main_slices,
            "feature_importance": res_main["feature_importance"],
        }

        (exp_dir / "metrics.json").write_text(json.dumps(exp_result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_results.append(exp_result)

        print(f"  -> Window 3 RMSPE: {main_metrics['rmspe']*100:.2f}% (MAE: {main_metrics['mae']:.1f}) | 3-Win RMSPE: {mean_rmspe*100:.2f}% ± {std_rmspe*100:.2f}% | 3-Win MAE: {mean_mae:.1f} ± {std_mae:.1f}", flush=True)
        if bootstrap_info and exp_id != "exp0_baseline_linear":
            print(f"  -> Rel. Impr. vs Baseline (W3): {bootstrap_info['relative_improvement_point']*100:.2f}% (95% CI: [{bootstrap_info['relative_improvement_95_ci'][0]*100:.2f}%, {bootstrap_info['relative_improvement_95_ci'][1]*100:.2f}%])", flush=True)

    # 4. Compute Bootstrap vs CatBoost Full for Ablations and Extended
    full_preds_dict = all_window_preds.get("exp4_catboost_full")
    pooled_val = pd.concat([window_datasets[w[0]][1] for w in eval_windows], ignore_index=True)
    if full_preds_dict:
        pooled_full = np.concatenate([full_preds_dict[w[0]] for w in eval_windows])
        for res in summary_results:
            if res["id"] in all_window_preds:
                cand_dict = all_window_preds[res["id"]]
                pooled_cand = np.concatenate([cand_dict[w[0]] for w in eval_windows])
                res["bootstrap_vs_full_3w"] = bootstrap_delta_rmspe(
                    pooled_val, pooled_full, pooled_cand, samples=config.get("bootstrap_samples", 1000), seed=seed
                )
                res["bootstrap_vs_full_w3"] = bootstrap_delta_rmspe(
                    main_val_df, full_preds_dict["window_3_main"], cand_dict["window_3_main"], samples=config.get("bootstrap_samples", 1000), seed=seed
                )

    # Save summary registry
    summary_path = Path("artifacts/experiments_summary.json")
    summary_path.write_text(json.dumps(summary_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate Markdown Tables: Separated Experiments and Ablations
    experiments_list = [r for r in summary_results if not r["id"].startswith("ablation")]
    ablations_list = [r for r in summary_results if r["id"].startswith("ablation") or r["id"] in ["exp4_catboost_full", "exp3_features_linear"]]

    table_lines = [
        "# Сводные результаты (ДЗ 2)",
        "",
        "## 1. Таблица экспериментов (сравнение моделей с baseline)",
        "",
        "| Эксперимент | Модель | Число признаков | 3 окна RMSPE | 3 окна MAE | Изменение к baseline (3 окна) | 95% интервал (3 окна) | RMSPE (окно 3, справка) | Итог |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for res in experiments_list:
        w3_rmspe = res["main_window"]["rmspe"] * 100
        mw_rmspe_mean = res["multi_window"]["mean_rmspe"] * 100
        mw_rmspe_std = res["multi_window"]["std_rmspe"] * 100
        mw_mae_mean = res["multi_window"]["mean_mae"]
        mw_mae_std = res["multi_window"]["std_mae"]
        boot = res.get("bootstrap_vs_baseline_3w") or {}
        rel_imp = boot.get("relative_improvement_point", 0.0)
        ci = boot.get("relative_improvement_95_ci", [0.0, 0.0])
        excludes_zero = boot.get("excludes_zero", False)

        if res["id"] == "exp0_baseline_linear":
            delta_str = "0.00%"
            ci_str = "-"
            status = "Базовая модель"
        else:
            delta_str = f"{-rel_imp * 100:+.2f}%"
            ci_low = -ci[1] * 100
            ci_high = -ci[0] * 100
            ci_str = f"[{ci_low:+.2f}%, {ci_high:+.2f}%]"
            if excludes_zero and rel_imp > 0:
                status = "Интервал ниже нуля"
            elif excludes_zero and rel_imp < 0:
                status = "Интервал выше нуля"
            else:
                status = "Включает ноль"

        table_lines.append(
            f"| {res['id']} | {res['model_type']} | {res['feature_count']} | {mw_rmspe_mean:.2f} ± {mw_rmspe_std:.2f}% | {mw_mae_mean:.1f} ± {mw_mae_std:.1f} | {delta_str} | {ci_str} | {w3_rmspe:.2f}% | {status} |"
        )

    table_lines.extend([
        "",
        "## 2. Таблица абляций (вклад компонентов в CatBoost Full)",
        "",
        "| Вариант | Что отключено / изменено | Число признаков | 3 окна RMSPE | 3 окна MAE | Изменение RMSPE к Full (3 окна) | 95% интервал к Full (3 окна) | RMSPE (окно 3, справка) | Вывод |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])

    full_mw_rmspe = catboost_full_metrics["mean_rmspe"] * 100 if catboost_full_metrics else 15.11
    ablation_descriptions = {
        "exp4_catboost_full": ("Полная модель (все признаки)", "Ориентир (наилучшее качество)"),
        "ablation1_no_store_meta": ("Без store.csv (формат, ассортимент, конкуренты)", "Метаданные формата и конкурентов помогают модели"),
        "ablation2_no_promo_split": ("Без разделения промо-истории", "RMSPE почти не изменился, но MAE вырос на днях без акций"),
        "ablation3_no_calendar": ("Без календаря (день месяца, зарплатные даты)", "Ошибки заметно выше; календарь стабилизирует прогноз во времени"),
        "exp3_features_linear": ("Замена модели: Ridge на том же полном наборе", "Линейная форма уступает бустингу, вероятно, из-за нелинейных эффектов и корреляций"),
    }

    for res in ablations_list:
        w3_rmspe = res["main_window"]["rmspe"] * 100
        mw_rmspe_mean = res["multi_window"]["mean_rmspe"] * 100
        mw_rmspe_std = res["multi_window"]["std_rmspe"] * 100
        mw_mae_mean = res["multi_window"]["mean_mae"]
        mw_mae_std = res["multi_window"]["std_mae"]

        what_disabled, conclusion = ablation_descriptions.get(res["id"], (res["name"], "-"))
        boot_full = res.get("bootstrap_vs_full_3w") or {}
        diff_ci = boot_full.get("delta_rmspe_95_ci", [0.0, 0.0])
        boot_delta = boot_full.get("delta_rmspe_point", 0.0) * 100
        delta_full_str = "0.00 п.п." if res["id"] == "exp4_catboost_full" else f"{boot_delta:+.2f} п.п."
        diff_ci_str = f"[{diff_ci[0]*100:+.2f} п.п., {diff_ci[1]*100:+.2f} п.п.]" if res["id"] != "exp4_catboost_full" else "-"

        table_lines.append(
            f"| {res['id']} | {what_disabled} | {res['feature_count']} | {mw_rmspe_mean:.2f} ± {mw_rmspe_std:.2f}% | {mw_mae_mean:.1f} ± {mw_mae_std:.1f} | {delta_full_str} | {diff_ci_str} | {w3_rmspe:.2f}% | {conclusion} |"
        )

    table_path = Path("artifacts/experiments_table.md")
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"\nAll experiments finished! Summary saved to {summary_path} and {table_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rossmann HW2 experiment suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/hw2.json"))
    parser.add_argument("--only-exp", type=str, default=None, help="Run only specified experiment id")
    args = parser.parse_args()
    run_all(args.config, only_exp=args.only_exp)
