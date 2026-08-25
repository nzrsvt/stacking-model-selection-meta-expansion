import os
import random
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DEFAULT_N_PARTS_RANGE = (2, 8)

@dataclass
class Segment:
    from_: float
    to_: float

    def __repr__(self) -> str:
        return f"[{self.from_:.3f}, {self.to_:.3f}]"


def piecewise_linear_transform(X: np.ndarray, intervals: List[float]) -> np.ndarray:
    n_samples, n_features = X.shape
    n_parts = len(intervals) - 1
    if intervals[0] != 0 or intervals[-1] != 1:
        raise ValueError("Інтервали мають починатися з 0 і закінчуватися 1")
    if not all(intervals[i] < intervals[i + 1] for i in range(len(intervals) - 1)):
        raise ValueError("Інтервали мають бути впорядковані за зростанням")

    X_exp = np.zeros((n_samples, n_features * n_parts))
    for j in range(n_features):
        x_j = X[:, j]
        for s in range(n_parts):
            a_prev, a_curr = intervals[s], intervals[s + 1]
            idx = j * n_parts + s
            X_exp[x_j <= a_prev, idx] = 0
            mask = (x_j > a_prev) & (x_j <= a_curr)
            X_exp[mask, idx] = x_j[mask] - a_prev
            X_exp[x_j > a_curr, idx] = a_curr - a_prev
    return X_exp


def intervals_to_segments(intervals: List[float]) -> List[Segment]:
    return [Segment(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)]


def calculate_metrics(y_true, y_pred, stage_name: str = "") -> dict:
    return {
        f"{stage_name}MAPE":      mean_absolute_percentage_error(y_true, y_pred),
        f"{stage_name}RMSE":      np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{stage_name}MAE":       mean_absolute_error(y_true, y_pred),
        f"{stage_name}Max Error": max_error(y_true, y_pred),
        f"{stage_name}Median AE": median_absolute_error(y_true, y_pred),
        f"{stage_name}MSE":       mean_squared_error(y_true, y_pred),
        f"{stage_name}R2":        r2_score(y_true, y_pred),
    }


def load_dataset(csv_file: str, target_col: str = None, drop_cols=None):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    if target_col is None:
        target_col = df.columns[-1]

    if target_col not in df.columns:
        raise ValueError(
            f"Цільова змінна '{target_col}' відсутня у датасеті.\n"
            f"Доступні стовпці: {list(df.columns)}"
        )

    cols_to_drop = [c for c in (drop_cols or []) if c != target_col]
    X = df.drop(columns=[target_col] + cols_to_drop)

    try:
        y = pd.to_numeric(df[target_col], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Цільова змінна '{target_col}' не може бути перетворена на числову: {exc}"
        ) from exc

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"  Датасет    : '{csv_file}'")
    print(f"  Форма      : {df.shape[0]} рядків × {df.shape[1]} стовпців")
    print(f"  Ціль       : '{target_col}'  (тип: {y.dtype})")
    print(f"  Ознаки     : {X.shape[1]} всього  "
          f"({len(num_cols)} числових, {len(cat_cols)} категоріальних)")
    if cat_cols:
        print(f"  Категоріальні: {cat_cols}")
    if drop_cols:
        print(f"  Видалено   : {cols_to_drop}")

    return X, y


def _make_preprocessor():
    return make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            make_column_selector(dtype_include=["object", "category"]),
        ),
        remainder="drop",
    )


def _to_array(v) -> np.ndarray:
    return v.values if hasattr(v, "values") else np.asarray(v)


def _unique_model_names(models: list) -> List[str]:
    def _get_name(m):
        if hasattr(m, "steps"):         
            return type(m.steps[-1][1]).__name__
        return type(m).__name__

    base_names = [_get_name(m) for m in models]
    totals = Counter(base_names)
    seen: Counter = Counter()
    result = []
    for name in base_names:
        if totals[name] > 1:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            result.append(name)
    return result


def _autofit_excel(writer):
    for ws in writer.sheets.values():
        for col in ws.columns:
            col = list(col)
            width = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)

def calibrate_thresholds(
    diversity_df: pd.DataFrame,
    top_n_pairs: int = 8,
) -> dict:
    n = len(diversity_df)
    if n == 0:
        raise ValueError("diversity_df порожній")

    k = min(top_n_pairs, n)

    def _worst_threshold(series: pd.Series, higher_is_worse: bool) -> float:
        s = series.dropna()
        if len(s) == 0:
            return 0.0
        sorted_vals = s.sort_values(ascending=not higher_is_worse).values
        idx = min(k - 1, len(sorted_vals) - 1)
        return float(sorted_vals[idx])

    rho_pred_max = _worst_threshold(diversity_df["rho_pred"], higher_is_worse=True)

    rho_err_bad  = _worst_threshold(diversity_df["rho_err"],  higher_is_worse=True)

    med_norm_min = _worst_threshold(diversity_df["MED_norm"], higher_is_worse=False)

    div_max      = _worst_threshold(diversity_df["Div"],      higher_is_worse=True)

    rho_err_ok    = rho_err_bad  * (0.5  / 0.75)   
    rho_pred_soft = rho_pred_max * (0.8  / 0.935)  
    div_soft      = div_max      * (0.4  / 0.36)   

    thresholds = {
        "rho_pred_max":  rho_pred_max,
        "rho_pred_soft": rho_pred_soft,
        "rho_err_bad":   rho_err_bad,
        "rho_err_ok":    rho_err_ok,
        "med_norm_min":  med_norm_min,
        "div_max":       div_max,
        "div_soft":      div_soft,
        "n_total_pairs": n,
        "top_n_pairs":   k,
    }

    sep = "-" * 54
    print(f"\n  {sep}")
    print("  АВТОКАЛІБРУВАННЯ ПОРОГІВ РІЗНОРІДНОСТІ")
    print(f"  Всього пар : {n}  |  top_n_pairs (відсіюємо гірших) = {k}")
    print(f"  {sep}")
    rho_pred_vals = diversity_df["rho_pred"].dropna().sort_values(ascending=False).values
    rho_err_vals  = diversity_df["rho_err"].dropna().sort_values(ascending=False).values
    med_norm_vals = diversity_df["MED_norm"].dropna().sort_values(ascending=True).values
    div_vals      = diversity_df["Div"].dropna().sort_values(ascending=False).values
    print(f"  rho_pred  : топ-{k} найбільших → [{', '.join(f'{v:.4f}' for v in rho_pred_vals[:k])}]")
    print(f"              X якщо >= {rho_pred_max:.4f}  "
          f"(залишиться {int(sum(diversity_df['rho_pred'].fillna(0) < rho_pred_max))}/{n} пар)")
    print(f"  rho_err   : топ-{k} найбільших → [{', '.join(f'{v:.4f}' for v in rho_err_vals[:k])}]")
    print(f"              X якщо > {rho_err_bad:.4f}  |  OK якщо <= {rho_err_ok:.4f}")
    print(f"  MED_norm  : топ-{k} найменших  → [{', '.join(f'{v:.4f}' for v in med_norm_vals[:k])}]")
    print(f"              OK якщо > {med_norm_min:.4f}")
    print(f"  Div       : топ-{k} найбільших → [{', '.join(f'{v:.4f}' for v in div_vals[:k])}]")
    print(f"              X якщо >= {div_max:.4f}  "
          f"(залишиться {int(sum(diversity_df['Div'].fillna(0) < div_max))}/{n} пар)")
    print(f"  {sep}")

    return thresholds


def is_pair_recommended(
    rho_pred, rho_err, med_norm, div,
    r2_i=None, r2_j=None,
    thresholds: dict = None,
) -> str:
    """
    Оцінює пару моделей за метриками різнорідності.

    Якщо передано thresholds (результат calibrate_thresholds) —
    використовуються адаптивні дата-залежні пороги.
    Інакше — захардкоджені значення за замовчуванням.
    """
    if thresholds is not None:
        _rho_pred_max = thresholds["rho_pred_max"]
        _rho_err_bad  = thresholds["rho_err_bad"]
        _rho_err_ok   = thresholds["rho_err_ok"]
        _med_norm_min = thresholds["med_norm_min"]
        _div_max      = thresholds["div_max"]
        _rho_pred_soft = thresholds["rho_pred_soft"]
        _div_soft      = thresholds["div_soft"]
    else:
        _rho_pred_max = 0.935
        _rho_err_bad  = 0.75
        _rho_err_ok   = 0.5
        _med_norm_min = 0.25
        _div_max      = 0.36
        _rho_pred_soft = 0.8
        _div_soft      = 0.4

    if r2_i is not None and r2_j is not None:
        if r2_i < 0.3 or r2_j < 0.3:
            return "X Погана якість моделі"

    if rho_pred > _rho_pred_max:
        return f"X Надмірна схожість прогнозів (rho_pred > {_rho_pred_max:.3f})"

    if div > _div_max:
        return f"X Низька різнорідність (Div > {_div_max:.3f})"

    if rho_err > _rho_err_bad:
        return f"X Висока rho_err > {_rho_err_bad:.3f}"

    if rho_err < 0 and med_norm >= _med_norm_min and div <= _div_max:
        return "OK Відмінно (rho_err < 0, MED > поріг, Div < поріг)"

    if rho_pred <= _rho_pred_max and rho_err <= _rho_err_ok and med_norm >= _med_norm_min and div <= _div_max:
        return "OK Рекомендовано (всі критерії виконано)"

    if rho_err <= _rho_err_ok and med_norm >= _med_norm_min:
        return f"OK Прийнятно (rho_err <= {_rho_err_ok:.3f}, MED > поріг)"

    if rho_pred < _rho_pred_soft and div < _div_soft:
        return f"OK Прийнятно (rho_pred < {_rho_pred_soft:.3f}, Div < {_div_soft:.3f})"

    if med_norm < _med_norm_min and rho_err <= _rho_err_ok and div <= _div_max:
        return f"! Обережно (мала MED < {_med_norm_min:.3f})"

    if rho_err > _rho_err_ok:
        return f"! Обережно (висока rho_err > {_rho_err_ok:.3f})"

    return "~ Жоден з критичних критеріїв не порушено"


def calculate_diversity_metrics(
    predictions_dict: dict,
    y_true: np.ndarray,
    r2_scores: dict = None,
    thresholds: dict = None,
) -> pd.DataFrame:
    names = list(predictions_dict.keys())
    y_std = np.std(y_true)
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mi, mj = names[i], names[j]
            r2_i = r2_scores.get(mi, 0) if r2_scores else None
            r2_j = r2_scores.get(mj, 0) if r2_scores else None

            pi, pj = predictions_dict[mi], predictions_dict[mj]
            ei, ej = y_true - pi, y_true - pj

            rho_pred = float(np.corrcoef(pi, pj)[0, 1])
            rho_err  = float(np.corrcoef(ei, ej)[0, 1])
            med_norm = float(np.mean(np.abs(ei - ej)) / y_std)
            ei_c, ej_c = ei - ei.mean(), ej - ej.mean()
            denom = np.sum(ei_c ** 2) + np.sum(ej_c ** 2)
            div   = float(np.sum(ei_c * ej_c) / denom) if denom > 0 else 0.0

            rows.append({
                "Model_Pair":  f"{mi} - {mj}",
                "rho_pred":    rho_pred,
                "rho_err":     rho_err,
                "MED_norm":    med_norm,
                "MED_abs":     float(np.mean(np.abs(ei - ej))),
                "Div":         div,
                "Recommended": is_pair_recommended(
                    rho_pred, rho_err, med_norm, div, r2_i, r2_j,
                    thresholds=thresholds,
                ),
            })

    return pd.DataFrame(rows)


def _train_base_models_stage(
    X_train_sc, X_test_sc,
    y_train, y_test,
    base_models: list,
    model_names: List[str],
    kfold: KFold,
) -> dict:
    y_tr = _to_array(y_train)
    y_te = _to_array(y_test)

    train_preds:       dict = {}
    test_preds:        dict = {}
    train_r2:          dict = {}
    test_r2:           dict = {}
    train_metrics_all: dict = {}
    test_metrics_all:  dict = {}
    fitted_models:     dict = {}
    model_train_times: dict = {}

    for idx, (model, name) in enumerate(zip(base_models, model_names), 1):
        print(f"\n  [{idx}/{len(base_models)}] {name}")

        t_model_start = time.perf_counter()

        oof = cross_val_predict(
            model, X_train_sc, y_tr,
            cv=kfold, method="predict", n_jobs=-1,
        )
        train_preds[name] = oof
        train_r2[name]    = r2_score(y_tr, oof)

        m = deepcopy(model)
        m.fit(X_train_sc, y_tr)
        te_pred = m.predict(X_test_sc)
        test_preds[name]    = te_pred
        test_r2[name]       = r2_score(y_te, te_pred)
        fitted_models[name] = m

        train_metrics_all[name] = calculate_metrics(y_tr, oof,     "Train_OOF_")
        test_metrics_all[name]  = calculate_metrics(y_te, te_pred, "Test_")

        model_train_times[name] = time.perf_counter() - t_model_start

        print(f"    Train OOF R2: {train_r2[name]:.4f}  |  Test R2: {test_r2[name]:.4f}"
              f"  |  Час: {model_train_times[name]:.4f} сек")

    t_base_models = sum(model_train_times.values())
    print(f"\n  Сумарний час навчання всіх кандидатів: {t_base_models:.4f} сек")

    poor = [n for n, r in train_r2.items() if r < 0.3]
    if poor:
        print(f"\n  Моделі з R2 < 0.3 (буде виключено): {poor}")
    else:
        print("\n  Всі моделі мають R2 >= 0.3")

    diversity_train_raw = calculate_diversity_metrics(train_preds, y_tr, train_r2)

    return {
        "train_preds":         train_preds,
        "test_preds":          test_preds,
        "train_r2":            train_r2,
        "test_r2":             test_r2,
        "train_metrics_all":   train_metrics_all,
        "test_metrics_all":    test_metrics_all,
        "fitted_models":       fitted_models,
        "model_train_times":   model_train_times,
        "t_base_models":       t_base_models,
        "poor_models":         poor,
        "diversity_train_raw": diversity_train_raw,
        "y_tr":                y_tr,
        "y_te":                y_te,
    }


def _finalize_diversity_stage(
    trained: dict,
    top_n_pairs: int,
    csv_file: str,
) -> dict:
    thresholds = calibrate_thresholds(
        trained["diversity_train_raw"], top_n_pairs=top_n_pairs
    )
    diversity_train = calculate_diversity_metrics(
        trained["train_preds"], trained["y_tr"], trained["train_r2"], thresholds
    )
    diversity_test = calculate_diversity_metrics(
        trained["test_preds"], trained["y_te"], trained["test_r2"], thresholds
    )

    print(f"\n  Метрики різнорідності (Train OOF):")
    print(diversity_train[
        ["Model_Pair", "rho_pred", "rho_err", "MED_norm", "Div", "Recommended"]
    ].to_string(index=False))

    _save_diversity_excel(
        diversity_train, diversity_test,
        trained["train_r2"], trained["test_r2"],
        csv_file, thresholds,
    )

    return {
        "diversity_train":    diversity_train,
        "diversity_test":     diversity_test,
        "train_r2":           trained["train_r2"],
        "test_r2":            trained["test_r2"],
        "train_metrics_all":  trained["train_metrics_all"],
        "test_metrics_all":   trained["test_metrics_all"],
        "train_preds":        trained["train_preds"],
        "test_preds":         trained["test_preds"],
        "fitted_models":      trained["fitted_models"],
        "poor_models":        trained["poor_models"],
        "t_base_models":      trained["t_base_models"],
        "model_train_times":  trained["model_train_times"],
        "thresholds":         thresholds,
    }


def run_diversity_stage(
    X_train_sc, X_test_sc,
    y_train, y_test,
    base_models: list,
    model_names: List[str],
    kfold: KFold,
    csv_file: str,
    top_n_pairs: int,
) -> dict:
    sep = "=" * 60
    print(f"\n{sep}")
    print("КРОК 1 — АНАЛІЗ РІЗНОРІДНОСТІ БАЗОВИХ МОДЕЛЕЙ")
    print(f"  Кандидатів: {len(base_models)}  |  CV: {kfold.n_splits}-fold  |  top_n_pairs: {top_n_pairs}")
    print(sep)

    trained = _train_base_models_stage(
        X_train_sc, X_test_sc, y_train, y_test, base_models, model_names, kfold,
    )
    return _finalize_diversity_stage(trained, top_n_pairs, csv_file)


def _target_n_models(n_candidates: int) -> int:
    return max(2, round(n_candidates / 3))


def _silent_selection(trained: dict, top_n_pairs: int, min_r2: float) -> dict:
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        thresholds = calibrate_thresholds(
            trained["diversity_train_raw"], top_n_pairs=top_n_pairs
        )
        diversity_train = calculate_diversity_metrics(
            trained["train_preds"], trained["y_tr"],
            trained["train_r2"], thresholds,
        )
        sel_res = find_maximum_compatible_models(
            diversity_train, trained["train_r2"], min_r2=min_r2
        )
    return sel_res


def _search_top_n_pairs(
    trained:  dict,
    target_n: int,
    min_r2:   float = 0.3,
) -> int:
    n_total_pairs = len(trained["diversity_train_raw"])
    sep  = "=" * 60
    dash = "-" * 60

    print(f"\n{sep}")
    print("АВТОПІДБІР top_n_pairs")
    print(f"  Ціль        : {target_n} базових моделей")
    print(f"  Пар загалом : {n_total_pairs}")
    print(f"  Перебір     : top_n_pairs від 1 до {n_total_pairs}")
    print(f"  Навчання вже завершено — лише перекалібровуємо пороги.")
    print(sep)
    print(f"  {'top_n_pairs':>11}  {'відібрано':>9}  моделі")
    print(f"  {dash}")

    prev_n_sel  = None
    found_top_n = None

    for top_n in range(1, n_total_pairs + 1):
        sel_res  = _silent_selection(trained, top_n, min_r2)
        n_sel    = len(sel_res["selected_models"])
        selected = sel_res["selected_models"]

        if n_sel == target_n:
            marker = "  ← ЦІЛЬ ✓"
        elif n_sel < target_n and (prev_n_sel is None or prev_n_sel > target_n):
            marker = "  ← перша точка ≤ цілі"
        else:
            marker = ""

        print(f"  {top_n:>11}  {n_sel:>9}  {selected}{marker}")

        if n_sel == target_n and found_top_n is None:
            found_top_n = top_n

        if (n_sel < target_n
                and prev_n_sel is not None
                and prev_n_sel > target_n
                and found_top_n is None):
            found_top_n = top_n 

        if found_top_n is not None and top_n > found_top_n:
            break

        prev_n_sel = n_sel

    print(f"  {dash}")

    if found_top_n is not None:
        n_final = len(_silent_selection(trained, found_top_n, min_r2)["selected_models"])
        if n_final == target_n:
            print(f"\n  ✓ Знайдено точний збіг: top_n_pairs = {found_top_n}"
                  f"  →  {n_final} моделей")
        else:
            print(f"\n  ~ Точний збіг недосяжний."
                  f"  Обрано top_n_pairs = {found_top_n}"
                  f"  →  {n_final} моделей  (ціль: {target_n})")
    else:
        found_top_n = n_total_pairs
        n_final = len(_silent_selection(trained, found_top_n, min_r2)["selected_models"])
        print(f"\n  ~ Кількість відібраних моделей не опустилась до {target_n}."
              f"  Використовуємо top_n_pairs = {found_top_n} (максимальна фільтрація)"
              f"  →  {n_final} моделей.")

    print(sep)
    return found_top_n


def find_maximum_compatible_models(
    diversity_df: pd.DataFrame,
    r2_scores: dict = None,
    min_r2: float = 0.3,
) -> dict:
    all_models = sorted({
        m.strip()
        for pair in diversity_df["Model_Pair"]
        for m in pair.split(" - ")
    })

    good_models = all_models
    if r2_scores:
        good_models = [m for m in all_models if r2_scores.get(m, 0) >= min_r2]
        bad = [m for m in all_models if m not in good_models]
        if bad:
            print(f"\n  Виключено з R2 < {min_r2}: {bad}")

    conflicts = set()
    for _, row in diversity_df.iterrows():
        if "X " in row["Recommended"]:
            m1, m2 = [s.strip() for s in row["Model_Pair"].split(" - ")]
            if m1 in good_models and m2 in good_models:
                conflicts.add(tuple(sorted([m1, m2])))

    print(f"\n  Придатних моделей: {len(good_models)}  |  Конфліктних пар: {len(conflicts)}")

    def evaluate(subset) -> tuple:
        rec = exc = med_s = r2_s = cnt = 0
        for i, m1 in enumerate(subset):
            for m2 in subset[i + 1:]:
                mask = (
                    (diversity_df["Model_Pair"] == f"{m1} - {m2}") |
                    (diversity_df["Model_Pair"] == f"{m2} - {m1}")
                )
                row = diversity_df[mask]
                if not row.empty:
                    r = row.iloc[0]
                    if "OK " in r["Recommended"]:
                        rec += 1
                        if "Відмінно" in r["Recommended"]:
                            exc += 1
                    med_s += r["MED_norm"]
                    cnt += 1
        if r2_scores:
            r2_s = sum(r2_scores.get(m, 0) for m in subset) / len(subset)
        return (rec, med_s / cnt if cnt else 0.0, r2_s, exc)

    def has_conflict(subset) -> bool:
        return any(
            tuple(sorted([m1, m2])) in conflicts
            for i, m1 in enumerate(subset)
            for m2 in subset[i + 1:]
        )

    best_subset, best_quality = [], (0, 0.0, 0.0, 0)

    for size in range(len(good_models), 0, -1):
        print(f"  Перевірка розміру {size}...", end="", flush=True)
        valid = [s for s in combinations(good_models, size) if not has_conflict(s)]
        print(f" знайдено {len(valid)}")
        if valid:
            for s in valid:
                q = evaluate(s)
                if q > best_quality:
                    best_quality, best_subset = q, list(s)
            break

    excluded = [m for m in good_models if m not in best_subset]

    print(f"\n  Відібрано: {best_subset}")
    print(f"  Рекомендованих пар: {best_quality[0]}  |  Відмінних: {best_quality[3]}")
    print(f"  Середня MED_norm: {best_quality[1]:.4f}  |  Середня R2: {best_quality[2]:.4f}")

    return {
        "selected_models": best_subset,
        "excluded_models": excluded,
        "quality_metrics": best_quality,
    }


def compare_with_excluded_models(
    diversity_df: pd.DataFrame, selected: list, excluded: list
):
    if not excluded:
        print("\n  Всі придатні моделі увійшли до відібраної множини.")
        return
    print("\n  ПРИЧИНИ ВИКЛЮЧЕННЯ:")
    for exc in excluded:
        reasons = []
        for sel in selected:
            mask = (
                (diversity_df["Model_Pair"] == f"{exc} - {sel}") |
                (diversity_df["Model_Pair"] == f"{sel} - {exc}")
            )
            row = diversity_df[mask]
            if not row.empty and "X " in row.iloc[0]["Recommended"]:
                reasons.append(f"{sel}: {row.iloc[0]['Recommended']}")
        if reasons:
            print(f"  * {exc} конфліктує з — " + "; ".join(reasons))
        else:
            print(f"  * {exc} — виключено для максимізації множини")

class QuantileIntervalTransformer:

    def __init__(self, n_parts: int):
        if n_parts < 1:
            raise ValueError(f"n_parts має бути >= 1, отримано: {n_parts}")
        self.n_parts = n_parts
        self.intervals_: Optional[List[float]] = None

    def fit(self, X: np.ndarray, y=None) -> "QuantileIntervalTransformer":
        pooled = np.clip(np.asarray(X, dtype=float), 0.0, 1.0).ravel()
        qs = np.linspace(0.0, 1.0, self.n_parts + 1)
        knots = np.quantile(pooled, qs).astype(float)
        knots[0]  = 0.0
        knots[-1] = 1.0

        eps = 1e-6
        for i in range(1, len(knots)):
            if knots[i] <= knots[i - 1]:
                knots[i] = knots[i - 1] + eps

        if knots[-1] != 1.0:
            knots = np.linspace(0.0, 1.0, self.n_parts + 1)

        self.intervals_ = knots.tolist()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.intervals_ is None:
            raise RuntimeError("QuantileIntervalTransformer не навчено: викличте fit() спочатку")
        return piecewise_linear_transform(np.clip(np.asarray(X, dtype=float), 0.0, 1.0), self.intervals_)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class IntervalOptimizer:

    DEFAULT_N_PARTS_RANGE = (2, 8)

    def __init__(
        self,
        random_state: int = 42,
        n_parts_range: Tuple[int, int] = DEFAULT_N_PARTS_RANGE,
    ):
        if len(n_parts_range) != 2 or n_parts_range[0] < 1 or n_parts_range[1] < n_parts_range[0]:
            raise ValueError(
                f"n_parts_range має бути кортежем (min, max) з min >= 1 та max >= min, "
                f"отримано: {n_parts_range}"
            )
        self.random_state    = random_state
        self.n_parts_range   = tuple(n_parts_range)
        self.best_n_parts:   Optional[int]         = None
        self.best_intervals: Optional[List[float]] = None
        self.best_score:     float                 = float("-inf")
        self._log:           List[dict]            = []

    def optimize(
        self,
        meta_train_norm: np.ndarray,
        y_train,
        meta_test_norm:  np.ndarray = None,
        y_test           = None,
        meta_model       = None,
        cv:      int             = 5,
        kfold:   Optional[KFold] = None,
        verbose: bool            = True,
        **_ignored,
    ) -> Tuple[int, List[float], float]:
        if kfold is None:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        y_tr = _to_array(y_train)
        sep  = "=" * 60

        n_min, n_max = self.n_parts_range

        if verbose:
            print(f"\n{sep}")
            print("КРОК 3 — ПІДБІР МЕЖ ІНТЕРВАЛІВ (квантилі) + ВИБІР n_parts")
            print("  Межі    <- квантилі розподілу прогнозів (детерміновано, без пошуку)")
            print("  n_parts <- обирається за OOF R² на TRAIN (крос-валідація, leak-free)")
            print(f"  n_parts діапазон  : {n_min}-{n_max}")
            print(sep)

        for n_parts in range(n_min, n_max + 1):
            oof_score_k = self._oof_r2_quantile(n_parts, meta_train_norm, y_tr, meta_model, kfold)
            intervals_k = QuantileIntervalTransformer(n_parts).fit(meta_train_norm).intervals_

            self._log.append({
                "n_parts":   n_parts,
                "intervals": intervals_k,
                "oof_score": oof_score_k,
            })

            if verbose:
                print(f"  n_parts={n_parts:>2}:  OOF R2={oof_score_k:>10.6f}  "
                      f"intervals={[round(v, 4) for v in intervals_k]}")

            if oof_score_k > self.best_score:
                self.best_score     = oof_score_k
                self.best_n_parts   = n_parts
                self.best_intervals = intervals_k

        if verbose:
            print(f"\n{sep}")
            print("  ПІДСУМОК ПО ВСІХ n_parts:")
            print(f"  {'n_parts':>8}  {'OOF R2':>12}  Інтервали")
            for e in self._log:
                marker = "  <- ОБРАНО" if e["n_parts"] == self.best_n_parts else ""
                print(
                    f"  {e['n_parts']:>8}  {e['oof_score']:>12.6f}  "
                    f"{[round(v, 4) for v in e['intervals']]}{marker}"
                )
            print(f"\n  Найкращий n_parts  : {self.best_n_parts}  (за OOF R2)")
            print(f"  Межі інтервалів    : {[round(v, 4) for v in self.best_intervals]}")
            print(f"  Сегменти           : {intervals_to_segments(self.best_intervals)}")
            print(f"  OOF R2             : {self.best_score:.6f}")

        return self.best_n_parts, self.best_intervals, self.best_score

    def _oof_r2_quantile(
        self,
        n_parts:         int,
        meta_train_norm: np.ndarray,
        y_tr:            np.ndarray,
        meta_model,
        kfold:           KFold,
    ) -> float:
        try:
            oof_pred = np.full(len(y_tr), np.nan, dtype=float)
            for tr_idx, val_idx in kfold.split(meta_train_norm):
                qit  = QuantileIntervalTransformer(n_parts).fit(meta_train_norm[tr_idx])
                X_tr  = qit.transform(meta_train_norm[tr_idx])
                X_val = qit.transform(meta_train_norm[val_idx])

                m = deepcopy(meta_model)
                m.fit(X_tr, y_tr[tr_idx])
                oof_pred[val_idx] = m.predict(X_val)

            return r2_score(y_tr, oof_pred)
        except Exception:
            return -1e10

def run_stacking_stage(
    meta_train:     np.ndarray,
    meta_test:      np.ndarray,
    y_train,
    y_test,
    meta_model,
    best_intervals: List[float],
    kfold:          KFold,
    model_names:    List[str],
) -> dict:
    sep = "=" * 60
    n_parts = len(best_intervals) - 1
    K = meta_train.shape[1]
    n_pairs = K * (K - 1) // 2

    print(f"\n{sep}")
    print("КРОК 4 — НАВЧАННЯ МЕТАМОДЕЛІ")
    print(f"  n_parts={n_parts}  |  інтервали={[round(v, 4) for v in best_intervals]}")
    print(sep)

    y_tr = _to_array(y_train)
    y_te = _to_array(y_test)

    t0 = time.perf_counter()
    scaler   = MinMaxScaler(feature_range=(0, 1))
    mt_norm  = scaler.fit_transform(meta_train)
    mte_norm = np.clip(scaler.transform(meta_test), 0, 1)
    t_norm   = time.perf_counter() - t0

    print("\n  [БЕЗ розширення]")
    mm_no = deepcopy(meta_model)
    mm_no.fit(meta_train, y_tr)
    y_oof_no  = cross_val_predict(deepcopy(meta_model), meta_train, y_tr,
                                  cv=kfold, method="predict", n_jobs=-1)
    y_test_no = mm_no.predict(meta_test)
    m_tr_no   = calculate_metrics(y_tr, y_oof_no,  "Train_OOF_")
    m_te_no   = calculate_metrics(y_te, y_test_no, "Test_")
    print(f"  Train OOF R2: {m_tr_no['Train_OOF_R2']:.6f}  |  Test R2: {m_te_no['Test_R2']:.6f}")

    print(f"\n  [З кусково-лінійним розширенням  n_parts={n_parts}]")

    t0 = time.perf_counter()
    mt_exp  = piecewise_linear_transform(mt_norm,  best_intervals)
    mte_exp = piecewise_linear_transform(mte_norm, best_intervals)
    t_expand_pw = time.perf_counter() - t0

    print(f"  Ознак після розширення: {mt_exp.shape[1]}  (було {mt_norm.shape[1]})")

    t0 = time.perf_counter()
    mm_exp = deepcopy(meta_model)
    mm_exp.fit(mt_exp, y_tr)
    t_meta_pw = time.perf_counter() - t0

    y_oof_exp  = cross_val_predict(deepcopy(meta_model), mt_exp, y_tr,
                                   cv=kfold, method="predict", n_jobs=-1)
    y_test_exp = mm_exp.predict(mte_exp)
    m_tr_exp   = calculate_metrics(y_tr, y_oof_exp,  "Train_OOF_")
    m_te_exp   = calculate_metrics(y_te, y_test_exp, "Test_")
    print(f"  Train OOF R2: {m_tr_exp['Train_OOF_R2']:.6f}  |  Test R2: {m_te_exp['Test_R2']:.6f}")
    print(f"  Час розширення: {t_expand_pw:.4f} сек  |  Час fit метамоделі: {t_meta_pw:.4f} сек")

    r2_no   = m_te_no["Test_R2"]
    r2_exp  = m_te_exp["Test_R2"]

    print(f"\n  ПОРІВНЯННЯ (Test R2):")
    print(f"  {'БЕЗ розширення':50s}: {r2_no:.6f}")
    print(f"  {'Кусково-лінійне (n_parts=' + str(n_parts) + ')':50s}: "
          f"{r2_exp:.6f}  ({r2_exp - r2_no:+.6f})")

    print(f"\n  ЗАМІР ЧАСУ МЕТА-ІНФЕРЕНСУ ({len(y_te)} зразків):")

    t0 = time.perf_counter()
    _n = np.clip(scaler.transform(meta_test), 0, 1)
    _e = piecewise_linear_transform(_n, best_intervals)
    _  = mm_exp.predict(_e)
    t_infer_pw = time.perf_counter() - t0

    print(f"  Кусково-лінійний : {t_infer_pw:.4f} сек  |  {t_infer_pw / len(y_te) * 1000:.4f} мс/зразок")

    return {
        "meta_model_no_expand": mm_no,
        "meta_model_expanded":  mm_exp,
        "meta_scaler":          scaler,
        "no_expand":  {"train_metrics": m_tr_no,   "test_metrics": m_te_no},
        "expanded":   {"train_metrics": m_tr_exp,  "test_metrics": m_te_exp},
        "predictions": {
            "y_train":     y_tr,        "y_test":       y_te,
            "y_oof_no":    y_oof_no,    "y_test_no":    y_test_no,
            "y_oof_exp":   y_oof_exp,   "y_test_exp":   y_test_exp,
            "meta_train":  meta_train,  "meta_test":    meta_test,
        },
        "train_times": {
            "norm_sec":         t_norm,
            "expand_pw_sec":    t_expand_pw,
            "meta_pw_sec":      t_meta_pw,
        },
        "inference_times": {
            "piecewise_sec": t_infer_pw,
            "n_samples":     len(y_te),
        },
        "model_names":    model_names,
        "best_intervals": best_intervals,
        "n_parts":        n_parts,
    }

def _save_diversity_excel(
    diversity_train: pd.DataFrame,
    diversity_test:  pd.DataFrame,
    train_r2: dict,
    test_r2:  dict,
    csv_file: str,
    thresholds: dict = None,
) -> str:
    out = f"{Path(csv_file).stem}_diversity_analysis.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        diversity_train.to_excel(writer, sheet_name="Diversity Train", index=False)
        diversity_test.to_excel(writer,  sheet_name="Diversity Test",  index=False)
        pd.DataFrame({
            "Model":    list(train_r2.keys()),
            "Train_R2": list(train_r2.values()),
            "Test_R2":  list(test_r2.values()),
        }).to_excel(writer, sheet_name="R2 Scores", index=False)

        if thresholds is not None:
            df_thr = pd.DataFrame([
                {
                    "Параметр":   "rho_pred_max",
                    "Значення":   thresholds["rho_pred_max"],
                    "Пояснення":  "X якщо rho_pred >= цього (надмірна схожість)",
                },
                {
                    "Параметр":   "rho_err_bad",
                    "Значення":   thresholds["rho_err_bad"],
                    "Пояснення":  "X якщо rho_err > цього (висока кореляція помилок)",
                },
                {
                    "Параметр":   "rho_err_ok",
                    "Значення":   thresholds["rho_err_ok"],
                    "Пояснення":  "OK якщо rho_err <= цього (прийнятна кореляція)",
                },
                {
                    "Параметр":   "med_norm_min",
                    "Значення":   thresholds["med_norm_min"],
                    "Пояснення":  "OK якщо MED_norm > цього (достатня різниця)",
                },
                {
                    "Параметр":   "div_max",
                    "Значення":   thresholds["div_max"],
                    "Пояснення":  "X якщо Div >= цього (низька різнорідність)",
                },
                {
                    "Параметр":   "top_n_pairs",
                    "Значення":   thresholds["top_n_pairs"],
                    "Пояснення":  "Кількість 'топових' пар для калібрування порогів",
                },
                {
                    "Параметр":   "n_total_pairs",
                    "Значення":   thresholds["n_total_pairs"],
                    "Пояснення":  "Всього пар проаналізовано",
                },
            ])
            df_thr.to_excel(writer, sheet_name="Adaptive Thresholds", index=False)

        _autofit_excel(writer)
    print(f"\n  -> Різнорідність збережено: '{out}'")
    return out


def _save_stacking_excel(
    stacking_res: dict,
    div_res:      dict,
    sel_res:      dict,
    optimizer:    IntervalOptimizer,
    csv_file:     str,
) -> str:
    out    = f"{Path(csv_file).stem}_stacking_results.xlsx"
    mnames = stacking_res["model_names"]
    preds  = stacking_res["predictions"]
    n_parts      = stacking_res["n_parts"]

    df_mt_train = pd.DataFrame({n: div_res["train_preds"][n] for n in mnames})
    df_mt_train["True_Value"] = preds["y_train"]
    df_mt_test  = pd.DataFrame({n: div_res["test_preds"][n] for n in mnames})
    df_mt_test["True_Value"]  = preds["y_test"]

    metric_keys = ["MAPE", "RMSE", "MAE", "Max Error", "Median AE", "MSE", "R2"]
    rows = []
    for stage, m_no, m_exp in [
        (
            "Train OOF",
            stacking_res["no_expand"]["train_metrics"],
            stacking_res["expanded"]["train_metrics"],
        ),
        (
            "Test",
            stacking_res["no_expand"]["test_metrics"],
            stacking_res["expanded"]["test_metrics"],
        ),
    ]:
        pfx = "Train_OOF_" if stage == "Train OOF" else "Test_"
        for mk in metric_keys:
            k = f"{pfx}{mk}"
            v_no   = m_no.get(k, "")
            v_exp  = m_exp.get(k, "")
            diff_exp  = (v_exp  - v_no) if isinstance(v_no, float) else ""
            rows.append({
                "Stage":  stage,
                "Metric": mk,
                "БЕЗ розширення":                                    v_no,
                f"Кусково-лінійне (n_parts={n_parts})":             v_exp,
                "Різниця (кусково-лінійне)":                         diff_exp,
            })
    df_comp = pd.DataFrame(rows)

    metric_keys_sel = ["MAPE", "RMSE", "MAE", "Max Error", "Median AE", "MSE", "R2"]
    sel_rows = []
    for m in mnames:
        row = {"Model": m}
        for mk in metric_keys_sel:
            row[f"Train_OOF_{mk}"] = div_res["train_metrics_all"].get(m, {}).get(f"Train_OOF_{mk}", "")
            row[f"Test_{mk}"]      = div_res["test_metrics_all"].get(m, {}).get(f"Test_{mk}", "")
        sel_rows.append(row)
    df_sel = pd.DataFrame(sel_rows)

    df_opt = pd.DataFrame([{
        "n_parts":      e["n_parts"],
        "Train_OOF_R2": e["oof_score"],
        "Intervals":    str([round(v, 4) for v in e["intervals"]]),
        "Segments":     str(intervals_to_segments(e["intervals"])),
        "Обрано":       "<- ОБРАНО (OOF R2)" if e["n_parts"] == optimizer.best_n_parts else "",
    } for e in optimizer._log])

    it = stacking_res["inference_times"]
    n  = it["n_samples"]
    df_infer = pd.DataFrame([
        {
            "Метод":            "Кусково-лінійний",
            "n_samples":        n,
            "Час_сек":          it["piecewise_sec"],
            "Час_мс_на_зразок": it["piecewise_sec"] / n * 1000,
        },
    ])

    mtt        = div_res["model_train_times"]
    tt         = stacking_res["train_times"]
    sel_names  = stacking_res["model_names"]

    t_selected = sum(mtt[n] for n in sel_names)

    col_pw   = "Кусково-лінійний, сек"

    total_pw   = t_selected + tt["norm_sec"] + tt["expand_pw_sec"]   + tt["meta_pw_sec"]

    per_model_rows = [
        {
            "Фаза":   f"  {n}",
            col_pw:   mtt[n],
        }
        for n in sel_names
    ]

    df_train_times = pd.DataFrame([
        {
            "Фаза":    "Відібрані базові моделі (спільне)",
            col_pw:    t_selected,
        },
        *per_model_rows,
        {
            "Фаза":    "Нормалізація (спільна)",
            col_pw:    tt["norm_sec"],
        },
        {
            "Фаза":    "Розширення ознак",
            col_pw:    tt["expand_pw_sec"],
        },
        {
            "Фаза":    "Навчання метамоделі",
            col_pw:    tt["meta_pw_sec"],
        },
        {
            "Фаза":    "РАЗОМ",
            col_pw:    total_pw,
        },
    ])

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_mt_train.to_excel(writer,    sheet_name="Meta Train",         index=False)
        df_mt_test.to_excel(writer,     sheet_name="Meta Test",          index=False)
        df_comp.to_excel(writer,        sheet_name="Metrics Comparison", index=False)
        df_sel.to_excel(writer,         sheet_name="Selected Models",    index=False)
        df_opt.to_excel(writer,         sheet_name="Interval Optimizer", index=False)
        df_train_times.to_excel(writer, sheet_name="Training Times",     index=False)
        df_infer.to_excel(writer,       sheet_name="Inference Times",    index=False)
        _autofit_excel(writer)

    print(f"\n  -> Результати стекінгу збережено: '{out}'")
    return out


def get_default_base_models() -> list:
    def with_imputer(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", model),
        ])

    def gp(kernel_base, **kwargs):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) * kernel_base
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-7, 1e1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42,
            **kwargs,
        )

    return [
        DecisionTreeRegressor(max_depth=None, min_samples_split=2, random_state=42),
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42),
        with_imputer(KNeighborsRegressor(n_neighbors=5, weights="uniform")),
        with_imputer(KernelRidge(alpha=1.0, kernel="rbf")),
        with_imputer(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        with_imputer(gp(RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)))),
        with_imputer(gp(Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5))),
        with_imputer(gp(RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)))),
    ]


def run_full_pipeline(
    csv_file:       str,
    target_col:     str   = None,
    drop_cols              = None,
    base_models:    list  = None,
    meta_model            = None,
    cv:             int   = 5,
    random_state:   int   = 42,
    test_size:      float = 0.3,
    min_r2:         float = 0.3,
    top_n_pairs:    Optional[int]          = None,
    n_parts_range:  Tuple[int, int]        = DEFAULT_N_PARTS_RANGE,
    skip_selection: bool                   = False,
) -> dict:
    BIG = "=" * 70
    print(f"\n{BIG}")
    print("ПОВНИЙ ПАЙПЛАЙН СТЕКІНГОВОГО АНСАМБЛЮ")
    print(f"  Датасет: {csv_file}")
    print(BIG)

    if base_models is None:
        base_models = get_default_base_models()
    if meta_model is None:
        meta_model = Ridge(alpha=0.1, solver="auto")

    auto_top_n_pairs = top_n_pairs is None

    n_min, n_max = n_parts_range
    print(f"  n_parts_range: ({n_min}, {n_max})  "
          f"→ {n_max - n_min + 1} варіант(ів) n_parts (квантильні межі)")

    X, y = load_dataset(csv_file, target_col, drop_cols)
    tnp_str = "авто" if auto_top_n_pairs else str(top_n_pairs)
    print(f"\n  Базових кандидатів: {len(base_models)}  |  CV: {cv}-fold  |  top_n_pairs: {tnp_str}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = _make_preprocessor()
    X_train_sc   = preprocessor.fit_transform(X_train)
    X_test_sc    = preprocessor.transform(X_test)
    kfold        = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    print(f"  Train: {X_train_sc.shape[0]}  |  Test: {X_test_sc.shape[0]}")

    model_names = _unique_model_names(base_models)

    trained = _train_base_models_stage(
        X_train_sc, X_test_sc, y_train, y_test, base_models, model_names, kfold,
    )

    if auto_top_n_pairs:
        target_n   = _target_n_models(len(base_models))
        top_n_pairs = _search_top_n_pairs(trained, target_n, min_r2=min_r2)

    div_res = _finalize_diversity_stage(trained, top_n_pairs, csv_file)

    sep = "=" * 60
    print(f"\n{sep}")
    if skip_selection:
        print("КРОК 2 — ВІДБІР БАЗОВИХ МОДЕЛЕЙ  [ВИМКНЕНО — skip_selection=True]")
        print(sep)
        selected_names = [n for n in div_res["train_r2"]
                          if n not in div_res["poor_models"]]
        print(f"\n  Відбір пропущено. Використовуються всі моделі з R2 >= {min_r2}:")
        for n in selected_names:
            print(f"    - {n}  (R2={div_res['train_r2'][n]:.4f})")
        sel_res = {
            "selected_models": selected_names,
            "excluded_models": [],
            "quality_metrics": (0, 0.0, 0.0, 0),
        }
    else:
        print("КРОК 2 — ВІДБІР ОПТИМАЛЬНОЇ МНОЖИНИ БАЗОВИХ МОДЕЛЕЙ")
        print(sep)

        sel_res = find_maximum_compatible_models(
            div_res["diversity_train"], div_res["train_r2"], min_r2=min_r2
        )
        compare_with_excluded_models(
            div_res["diversity_train"],
            sel_res["selected_models"],
            sel_res["excluded_models"],
        )

        selected_names = sel_res["selected_models"]
        if len(selected_names) < 2:
            print("\n  Відібрано < 2 моделей — використовуємо всі з R2 >= min_r2.")
            selected_names = [n for n in div_res["train_r2"]
                              if n not in div_res["poor_models"]]

    meta_train = np.column_stack([div_res["train_preds"][n] for n in selected_names])
    meta_test  = np.column_stack([div_res["test_preds"][n]  for n in selected_names])
    print(f"\n  Метавибірка: {meta_train.shape[0]} x {meta_train.shape[1]}"
          f"  (моделі: {selected_names})")

    scaler_for_opt   = MinMaxScaler(feature_range=(0, 1))
    mt_norm_for_opt  = scaler_for_opt.fit_transform(meta_train)
    mte_norm_for_opt = np.clip(scaler_for_opt.transform(meta_test), 0, 1)

    optimizer = IntervalOptimizer(random_state=random_state, n_parts_range=n_parts_range)
    best_n_parts, best_intervals, best_oof_r2 = optimizer.optimize(
        meta_train_norm = mt_norm_for_opt,
        y_train         = y_train,
        meta_test_norm  = mte_norm_for_opt,
        y_test          = y_test,
        meta_model      = meta_model,
        cv              = cv,
        kfold           = kfold,
        verbose         = True,
    )

    stack_res = run_stacking_stage(
        meta_train, meta_test, y_train, y_test,
        meta_model, best_intervals, kfold, selected_names,
    )

    print(f"\n{sep}")
    print("ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    print(sep)
    _save_stacking_excel(stack_res, div_res, sel_res, optimizer, csv_file)

    mtt = div_res["model_train_times"]
    tt  = stack_res["train_times"]

    t_selected = sum(mtt[n] for n in selected_names)

    total_pw   = t_selected + tt["norm_sec"] + tt["expand_pw_sec"]   + tt["meta_pw_sec"]

    W = 87
    print(f"\n  ЧАС НАВЧАННЯ ВІДІБРАНИХ МОДЕЛЕЙ ({', '.join(selected_names)}):")
    for n in selected_names:
        print(f"    {n:<42}: {mtt[n]:.4f} сек")
    print(f"    {'Разом відібрані':<42}: {t_selected:.4f} сек  "
          f"(з {div_res['t_base_models']:.4f} сек на всіх кандидатів)")

    print(f"\n  ПОВНИЙ ЧАС НАВЧАННЯ (тільки відібрані базові моделі):")
    print(f"  {'Фаза':<45} {'Кусково-лін.':>12}")
    print(f"  {'-' * W}")
    print(f"  {'Відібрані базові моделі (спільне), сек':<45} {t_selected:>12.4f}")
    print(f"  {'Нормалізація (спільна), сек':<45} {tt['norm_sec']:>12.4f}")
    print(f"  {'Розширення ознак, сек':<45} {tt['expand_pw_sec']:>12.4f}")
    print(f"  {'Навчання метамоделі, сек':<45} {tt['meta_pw_sec']:>12.4f}")
    print(f"  {'-' * W}")
    print(f"  {'РАЗОМ, сек':<45} {total_pw:>12.4f}")

    m_te_no   = stack_res["no_expand"]["test_metrics"]
    m_te_exp  = stack_res["expanded"]["test_metrics"]

    print(f"\n{BIG}")
    print("ПІДСУМОК ПАЙПЛАЙНУ")
    print(f"  Відібрані моделі       : {selected_names}")
    print(f"  n_parts_range          : {n_parts_range}")
    print(f"  Обраний n_parts        : {best_n_parts}  (за OOF R2 на трейні)")
    print(f"  Межі інтервалів        : {[round(v, 4) for v in best_intervals]}")
    print(f"  Сегменти               : {intervals_to_segments(best_intervals)}")
    
    for m in ["MAPE", "RMSE", "MAE", "Max Error", "Median AE", "MSE", "R2"]:
        v_no = m_te_no[f"Test_{m}"]
        v_exp = m_te_exp[f"Test_{m}"]
        print(f"  {'Test ' + m + ' БЕЗ розширення':50s}: {v_no:.6f}")
        print(f"  {'Test ' + m + ' кусково-лінійне (n_parts=' + str(best_n_parts) + ')':50s}: "
              f"{v_exp:.6f}  ({v_exp - v_no:+.6f})")

    thr = div_res["thresholds"]
    print(f"\n  АДАПТИВНІ ПОРОГИ (top_n_pairs={thr['top_n_pairs']}, з {thr['n_total_pairs']} пар):")
    print(f"  rho_pred_max : {thr['rho_pred_max']:.4f}")
    print(f"  rho_err_bad  : {thr['rho_err_bad']:.4f}  |  rho_err_ok: {thr['rho_err_ok']:.4f}")
    print(f"  med_norm_min : {thr['med_norm_min']:.4f}")
    print(f"  div_max      : {thr['div_max']:.4f}")
    print(BIG)

    return {
        "diversity":      div_res,
        "selection":      sel_res,
        "optimizer":      optimizer,
        "stacking":       stack_res,
        "best_n_parts":   best_n_parts,
        "best_intervals": best_intervals,
        "thresholds":     thr,
    }

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "insurance.csv"

    results = run_full_pipeline(
        csv_file      = csv_file,
        skip_selection=True
    )