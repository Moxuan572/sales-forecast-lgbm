#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle MMM 数据集 —— 周度销售额预测（LightGBM 回归）

流程:
  1. 数据清洗: 日期解析/排序/去重; 渠道花费缺失按 0 处理; 缺失目标值的样本剔除
  2. 特征工程: 时间特征、促销/节假日、渠道花费、滞后与滚动特征(严格 shift>=1)
  3. 防泄漏: 特征全部只使用 t-1 及更早的信息; 按时间顺序切分, 不打乱
  4. 训练评估: LightGBM 回归, 输出 MAE / R²; 附带滚动时序交叉验证
  5. 全量重训并导出 sales_forecast_lgbm.pkl

用法:
  python3 train_model.py [数据文件路径]
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


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    """读取数据并完成基础清洗。"""
    df = pd.read_csv(path, encoding="utf-8-sig")  # 去除 BOM
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.sort_values("date").drop_duplicates().reset_index(drop=True)

    # 渠道花费缺失: 缺失周的销售额显著低于有花费的周(如 google_ads:
    # 208 万 vs 247 万), 说明缺失代表"该渠道当周未投放", 因此填 0。
    spend_missing = df[SPEND_COLS].isna().sum().astype(int).to_dict()
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0.0)

    # 目标缺失的样本无法用于监督学习, 在特征构建后删除
    target_missing = int(df["sales"].isna().sum())

    report = {
        "rows_raw": int(len(df)),
        "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        "spend_missing_filled_with_zero": spend_missing,
        "target_missing_dropped": target_missing,
    }
    return df, report


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建特征。所有滞后/滚动特征只使用当前周之前的数据, 避免泄漏。"""
    d = df.copy()

    # --- 营销花费特征 ---
    d[SPEND_COLS] = d[SPEND_COLS].fillna(0.0)
    d["total_spend"] = d[SPEND_COLS].sum(axis=1)
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
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    print("=" * 62)
    print("1) 读取与清洗数据")
    print("=" * 62)
    raw, clean_report = load_and_clean(data_path)
    print(f"原始样本数: {clean_report['rows_raw']}")
    print(f"日期范围  : {clean_report['date_range'][0]} ~ {clean_report['date_range'][1]}")
    print(f"目标缺失丢弃: {clean_report['target_missing_dropped']} 行")
    print(f"渠道花费缺失填 0: {clean_report['spend_missing_filled_with_zero']}")

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
