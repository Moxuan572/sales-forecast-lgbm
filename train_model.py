#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle MMM 数据集 —— 周度销售额预测（LightGBM 回归）

流程:
  1. 数据清洗: 日期解析/排序/去重; 渠道花费文本 "Nan" 保留为 NaN(技术性缺失), 真正空白填 0(结构性无投放); 缺失目标值的样本剔除
  2. 特征工程: 时间特征、促销/节假日、渠道花费、滞后与滚动特征(严格 shift>=1)
  3. 防泄漏: 特征全部只使用 t-1 及更早的信息; 按时间顺序切分, 不打乱
  4. 训练评估: LightGBM 回归, 输出 MAE / R²; 附带滚动时序交叉验证
  5. 全量重训并导出 sales_forecast_lgbm.pkl

用法:
  python3 train_model.py [数据文件路径] [--zero-fill-spends | --keep-spend-nan]
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "mmm_dataset.csv"
MODEL_PATH = BASE_DIR / "sales_forecast_lgbm.pkl"
METADATA_PATH = BASE_DIR / "sales_forecast_lgbm_metadata.json"

SPEND_COLS = [
    "competitor_spend",
    "instagram_spend",
    "google_ads_spend",
    "tv_spend",
    "youtube_spend",
    "newspaper_spend",
    "influencer_spend",
    "ott_spend",
]
PROMO_CATEGORIES = ["BOGO", "Coupons", "Flash Sale", "Normal"]
SALES_LAGS = (1, 2, 3, 4)
TEST_WEEKS = 42  # 末尾约 20% 作为时间外测试集

LGBM_PARAMS = dict(
    objective="regression",  # L2 损失, 与 R² 指标一致
    n_estimators=200,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=5,
    colsample_bytree=0.8,
    subsample=0.7,
    subsample_freq=1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=1,  # 单线程保证可复现
    verbose=-1,
)


def read_dataset(path: Path) -> pd.DataFrame:
    """按扩展名读取 CSV / Excel，并统一列名。"""
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")  # 去除 BOM
    df.columns = [c.strip() for c in df.columns]
    return df


def load_and_clean(
    path: Path,
    zero_fill_spends: bool = False,
    blank_as_zero: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """读取数据并完成基础清洗。

    花费缺失分两类:
      - 文本 "Nan"/"None"/"null": 显式技术性缺失标记, 保留 NaN;
      - 真正的空白单元格 / 空字符串: 结构性无投放, 填 0。

    zero_fill_spends=True: 所有花费缺失都填 0 (兼容旧流程);
    blank_as_zero=True:    只把真正的空白填 0, 文本 "Nan" 保留 NaN;
    两者都为 False:        所有花费缺失都保留 NaN。
    """
    df = read_dataset(path)

    # date 可能是字符串(03-01-2022)或 Excel 日期单元格
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.sort_values("date").drop_duplicates().reset_index(drop=True)

    # 先区分“文本技术性缺失”与“真正空白”，再数值化。
    # 这样 fillna(0) 不会把文本 "Nan" 也当成结构性 0。
    technical_missing = pd.DataFrame(False, index=df.index, columns=SPEND_COLS)
    blank_missing = pd.DataFrame(False, index=df.index, columns=SPEND_COLS)
    for col in SPEND_COLS:
        values = df[col]
        if values.dtype == object:
            stripped = values.map(lambda v: v.strip() if isinstance(v, str) else v)
            technical_missing[col] = stripped.map(
                lambda v: isinstance(v, str) and v.lower() in {"nan", "none", "null"}
            )
            blank_missing[col] = stripped.map(
                lambda v: isinstance(v, str) and v == ""
            )
            values = stripped.where(~blank_missing[col], np.nan)
        numeric = pd.to_numeric(values, errors="coerce")
        # 非空但无法解析的文本也按技术性缺失处理，避免被误填成 0。
        technical_missing[col] = technical_missing[col] | (values.notna() & numeric.isna())
        df[col] = numeric
    # 非文本列中的 NaN（Excel 真空白）也归入 blank_missing
    blank_missing = blank_missing | (df[SPEND_COLS].isna() & ~technical_missing)

    # 目标/其他字段也做数值化与去空格
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["holiday"] = pd.to_numeric(df["holiday"], errors="coerce").astype(int)
    df["sales_promotion"] = df["sales_promotion"].astype(str).str.strip()

    # 渠道花费: 0 = 结构性无投放; NaN = 技术性缺失(采集/入库失败)。
    spend_missing = df[SPEND_COLS].isna().sum().astype(int).to_dict()
    spend_zero_in_source = (df[SPEND_COLS] == 0).sum().astype(int).to_dict()
    technical_missing_counts = technical_missing.sum().astype(int).to_dict()
    blank_missing_counts = blank_missing.sum().astype(int).to_dict()

    if zero_fill_spends:
        df[SPEND_COLS] = df[SPEND_COLS].fillna(0.0)
        policy = "fill_zero"
    elif blank_as_zero:
        df[SPEND_COLS] = df[SPEND_COLS].mask(blank_missing, 0.0)
        policy = "blank_zero_keep_text_nan"
    else:
        policy = "keep_as_nan"
    df[SPEND_COLS] = df[SPEND_COLS].astype("float64")
    spend_zero_after_policy = (df[SPEND_COLS] == 0).sum().astype(int).to_dict()
    spend_missing_after_policy = df[SPEND_COLS].isna().sum().astype(int).to_dict()

    # 目标缺失的样本无法用于监督学习, 在特征构建后删除
    target_missing = int(df["sales"].isna().sum())

    report = {
        "source_file": path.name,
        "rows_raw": int(len(df)),
        "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        "spend_missing_policy": policy,
        "spend_missing_counts": spend_missing,
        "spend_technical_missing_counts": technical_missing_counts,
        "spend_blank_cells_filled_zero": blank_missing_counts if blank_as_zero else {c: 0 for c in SPEND_COLS},
        "spend_missing_counts_after_policy": spend_missing_after_policy,
        "spend_zero_cells_in_source": spend_zero_in_source,
        "spend_zero_cells_after_policy": spend_zero_after_policy,
        "target_missing_dropped": target_missing,
    }
    return df, report


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建特征。所有滞后/滚动特征只使用当前周之前的数据, 避免泄漏。"""
    d = df.copy()

    # --- 营销花费特征 ---
    # 技术性缺失的渠道保留 NaN, 由 LightGBM 原生处理。
    # total_spend 只汇总已观测渠道; n_active_channels 只统计已观测且 >0 的渠道。
    d["total_spend"] = d[SPEND_COLS].sum(axis=1, skipna=True)
    d["n_active_channels"] = (d[SPEND_COLS] > 0).sum(axis=1)

    # --- 时间特征 ---
    iso = d["date"].dt.isocalendar()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["week"] = iso["week"].astype(int)
    d["quarter"] = d["date"].dt.quarter
    d["dayofyear"] = d["date"].dt.dayofyear
    d["time_index"] = np.arange(len(d))  # 线性趋势
    d["week_sin"] = np.sin(2 * np.pi * d["week"] / 52.18)  # 年周期
    d["week_cos"] = np.cos(2 * np.pi * d["week"] / 52.18)
    d["holiday"] = d["holiday"].astype(int)

    # --- 类别特征: 固定类别顺序, 保证训练/推理一致 ---
    d["sales_promotion"] = d["sales_promotion"].astype(
        pd.CategoricalDtype(categories=PROMO_CATEGORIES, ordered=False)
    )

    # --- 滞后特征 (用 shift(1) 起, 绝不把当周目标带入特征) ---
    sales = d["sales"]
    for lag in SALES_LAGS:
        d[f"sales_lag_{lag}"] = sales.shift(lag)

    # --- 滚动特征: 先 shift(1) 再 rolling, 窗口内不含当周 ---
    shifted_sales = sales.shift(1)
    d["sales_roll_4_mean"] = shifted_sales.rolling(4).mean()
    d["sales_roll_4_std"] = shifted_sales.rolling(4).std()
    d["sales_roll_8_mean"] = shifted_sales.rolling(8).mean()
    d["sales_roll_4_max"] = shifted_sales.rolling(4).max()
    d["sales_roll_4_min"] = shifted_sales.rolling(4).min()

    # --- 花费的滞后/滚动特征 ---
    d["total_spend_lag_1"] = d["total_spend"].shift(1)
    d["total_spend_roll_4_mean"] = d["total_spend"].shift(1).rolling(4).mean()
    for col in SPEND_COLS:
        d[f"{col}_lag_1"] = d[col].shift(1)

    return d


def prepare_xy(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """去掉目标缺失行, 返回特征矩阵和目标值。"""
    valid = features["sales"].notna()
    data = features.loc[valid]
    X = data.drop(columns=["date", "sales"])
    y = data["sales"].astype(float)
    return X, y


def make_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**LGBM_PARAMS)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"MAE  : {metrics['MAE']:,.0f}\n"
        f"R²   : {metrics['R2']:.4f}\n"
        f"RMSE : {metrics['RMSE']:,.0f}\n"
        f"MAPE : {metrics['MAPE_%']:.2f}%"
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    data_path = Path(args[0]) if args else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    # 默认:
    #   Excel: 真正空白 -> 0; 文本 "Nan"/"None"/"null" -> 保留 NaN
    #   CSV:   旧格式, 所有缺失 -> 0 (兼容旧流程)
    # 显式 --zero-fill-spends / --keep-spend-nan 可覆盖。
    is_excel = data_path.suffix.lower() in {".xlsx", ".xlsm"}
    if "--zero-fill-spends" in flags:
        zero_fill_spends = True
        blank_as_zero = False
    elif "--keep-spend-nan" in flags:
        zero_fill_spends = False
        blank_as_zero = False
    else:
        zero_fill_spends = not is_excel
        blank_as_zero = is_excel

    print("=" * 62)
    print("1) 读取与清洗数据")
    print("=" * 62)
    raw, clean_report = load_and_clean(
        data_path, zero_fill_spends=zero_fill_spends, blank_as_zero=blank_as_zero
    )
    print(f"数据文件  : {clean_report['source_file']}")
    print(f"原始样本数: {clean_report['rows_raw']}")
    print(f"日期范围  : {clean_report['date_range'][0]} ~ {clean_report['date_range'][1]}")
    print(f"目标缺失丢弃: {clean_report['target_missing_dropped']} 行")
    print(f"花费缺失策略: {clean_report['spend_missing_policy']}")
    spend_missing = clean_report["spend_missing_counts"]
    print(f"原始缺失花费单元格: {sum(spend_missing.values())} 个 -> {spend_missing}")
    blank_zero = clean_report["spend_blank_cells_filled_zero"]
    print(f"真正空白按 0 处理: {sum(blank_zero.values())} 个 -> {blank_zero}")
    tech_missing = clean_report["spend_technical_missing_counts"]
    print(f"文本技术性缺失(保留 NaN): {sum(tech_missing.values())} 个 -> {tech_missing}")
    final_missing = clean_report["spend_missing_counts_after_policy"]
    print(f"处理后仍为 NaN: {sum(final_missing.values())} 个 -> {final_missing}")
    print(
        "花费 0 值单元格: 源文件 "
        f"{sum(clean_report['spend_zero_cells_in_source'].values())} 个 -> 处理后 "
        f"{sum(clean_report['spend_zero_cells_after_policy'].values())} 个"
    )

    print("\n" + "=" * 62)
    print("2) 特征工程 (滞后特征严格 shift>=1, 防止数据泄漏)")
    print("=" * 62)
    features = build_features(raw)
    X_all, y_all = prepare_xy(features)
    print(f"可用样本数: {len(X_all)} | 特征数: {X_all.shape[1]}")
    print("滞后特征  :", [f"sales_lag_{l}" for l in SALES_LAGS])
    print("滚动特征  : sales_roll_4_mean/std, sales_roll_8_mean, sales_roll_4_max/min (均先 shift(1))")

    if len(X_all) <= TEST_WEEKS + 20:
        raise ValueError("样本量太少, 无法进行训练/测试切分")

    # --- 按时间顺序切分, 不打乱 ---
    X_train, y_train = X_all.iloc[:-TEST_WEEKS], y_all.iloc[:-TEST_WEEKS]
    X_test, y_test = X_all.iloc[-TEST_WEEKS:], y_all.iloc[-TEST_WEEKS:]
    test_dates = features.loc[X_test.index, "date"]
    train_dates = features.loc[X_train.index, "date"]

    print("\n" + "=" * 62)
    print("3) 训练与评估")
    print("=" * 62)
    print(f"训练集: {len(X_train)} 周 ({train_dates.min():%Y-%m-%d} ~ {train_dates.max():%Y-%m-%d})")
    print(f"测试集: {len(X_test)} 周 ({test_dates.min():%Y-%m-%d} ~ {test_dates.max():%Y-%m-%d})")

    # 滚动时序交叉验证(仅用训练集), 检查参数稳定性
    cv_maes, cv_r2s = [], []
    tscv = TimeSeriesSplit(n_splits=4)
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
        model = make_model()
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[va_idx])
        fold_mae = mean_absolute_error(y_train.iloc[va_idx], pred)
        fold_r2 = r2_score(y_train.iloc[va_idx], pred)
        cv_maes.append(fold_mae)
        cv_r2s.append(fold_r2)
        print(f"  CV 第 {fold} 折: MAE={fold_mae:,.0f}  R²={fold_r2:.3f}")
    print(f"  CV 平均   : MAE={np.mean(cv_maes):,.0f}  R²={np.mean(cv_r2s):.3f}")

    # 测试集评估
    eval_model = make_model()
    eval_model.fit(X_train, y_train)
    test_pred = eval_model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)

    print("\n测试集评估结果:")
    print(format_metrics(test_metrics))

    # 特征重要性 (gain)
    importance = pd.Series(
        eval_model.booster_.feature_importance(importance_type="gain"),
        index=X_train.columns,
    ).sort_values(ascending=False)
    print("\nTop 10 重要特征 (gain):")
    for name, gain in importance.head(10).items():
        print(f"  {name:32s} {gain:,.0f}")

    # --- 用全部数据重训, 导出最终模型 ---
    print("\n" + "=" * 62)
    print("4) 全量重训并导出模型")
    print("=" * 62)
    final_model = make_model()
    final_model.fit(X_all, y_all)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    # 校验持久化结果可正常预测
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    assert np.allclose(loaded_model.predict(X_all), final_model.predict(X_all))
    print(f"模型已保存: {MODEL_PATH}")

    metadata = {
        "model_file": MODEL_PATH.name,
        "task": "weekly sales forecasting (regression)",
        "target": "sales",
        "data": clean_report,
        "rows_used": int(len(X_all)),
        "features": list(X_all.columns),
        "categorical_features": ["sales_promotion"],
        "lag_features": [f"sales_lag_{l}" for l in SALES_LAGS],
        "leakage_guard": "all lag/rolling features use shift(1)+; chronological split; no shuffling",
        "train_period": [train_dates.min().strftime("%Y-%m-%d"), train_dates.max().strftime("%Y-%m-%d")],
        "test_period": [test_dates.min().strftime("%Y-%m-%d"), test_dates.max().strftime("%Y-%m-%d")],
        "params": {k: v for k, v in LGBM_PARAMS.items()},
        "cv": {"folds": 4, "mean_MAE": float(np.mean(cv_maes)), "mean_R2": float(np.mean(cv_r2s))},
        "holdout_metrics": test_metrics,
        "note": "holdout metrics are from the model trained on the first split; "
                "the exported model is refit on all labeled rows",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {METADATA_PATH}")
    print("\n完成 ✅")


if __name__ == "__main__":
    main()
