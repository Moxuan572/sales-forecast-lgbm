#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销售预测 FastAPI 服务

调用方只需提供原始业务字段:
  date, sales_promotion, holiday, 8 个渠道花费, sales_history (过去 4 周销售额)

服务端在 preprocess() 中自动生成训练时的 39 个特征:
  - 时间特征: year / month / week / quarter / dayofyear / time_index / week_sin / week_cos
  - 当周花费: total_spend / n_active_channels
  - 销售滞后: sales_lag_1~4 (来自 sales_history)
  - 销售滚动: sales_roll_4_mean/std/max/min, sales_roll_8_mean
  - 花费滞后: *_spend_lag_1 / total_spend_lag_1 / total_spend_roll_4_mean
    (入参不含花费历史, 采用"花费持续"假设: 用当周花费近似上周花费)

启动时:
  - joblib 加载 sales_forecast_lgbm.pkl
  - 读取 metadata, 严格校验特征顺序
  - 用一条示例请求跑通 preprocess, 确认输出列与 metadata 完全一致

接口:
  POST /predict  ->  {"predicted_sales": "2434879.82",
                      "predicted_roas": "3.96",
                      "action": "观察 (Observe)"}
  GET  /health   ->  健康检查
  GET  /docs     ->  Swagger UI

启动:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from datetime import date as date_cls
from pathlib import Path
from typing import Literal, get_args

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
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

# 训练数据从 2022-01-03 开始, time_index 以该周为 0
REFERENCE_MONDAY = pd.Timestamp("2022-01-03")
REFERENCE_DATE = date_cls(2022, 1, 3)


class PredictRequest(BaseModel):
    """单周预测请求(原始业务字段)。"""

    model_config = ConfigDict(extra="forbid")

    date: date_cls = Field(
        ..., description="待预测周的日期, ISO 格式, 例如 2026-10-20"
    )
    sales_promotion: Literal["BOGO", "Coupons", "Flash Sale", "Normal"] = Field(
        ..., description="促销类型"
    )
    holiday: Literal[0, 1] = Field(..., description="是否节假日: 0 或 1")

    competitor_spend: float = Field(..., ge=0, description="竞品花费")
    instagram_spend: float = Field(..., ge=0, description="Instagram 花费")
    google_ads_spend: float = Field(..., ge=0, description="Google Ads 花费")
    tv_spend: float = Field(..., ge=0, description="电视广告花费")
    youtube_spend: float = Field(..., ge=0, description="YouTube 花费")
    newspaper_spend: float = Field(..., ge=0, description="报纸广告花费")
    influencer_spend: float = Field(..., ge=0, description="达人营销花费")
    ott_spend: float = Field(..., ge=0, description="OTT 广告花费")

    sales_history: list[float] = Field(
        ...,
        min_length=4,
        max_length=8,
        description=(
            "过去各周销售额, 按时间从近到远: "
            "[t-1, t-2, t-3, t-4]。最少 4 个; "
            "若提供 8 个, sales_roll_8_mean 可精确计算, 否则该特征为缺失值"
        ),
    )

    @field_validator("date")
    @classmethod
    def _validate_date(cls, value: date_cls) -> date_cls:
        if value < REFERENCE_DATE:
            raise ValueError(f"date 不能早于训练数据起点 {REFERENCE_DATE.isoformat()}")
        return value

    @field_validator("sales_history")
    @classmethod
    def _validate_history(cls, value: list[float]) -> list[float]:
        if any((not math.isfinite(v)) or v < 0 for v in value):
            raise ValueError("sales_history 中的销售额必须是非负有限数值")
        return value

    @field_validator("*", mode="after")
    @classmethod
    def _reject_non_finite(cls, value, info):
        """拒绝 Infinity; NaN 同样无法通过 ge=0 约束。"""
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{info.field_name} 必须是有限数值")
        return value


Action = Literal["放量 (Scale Up)", "观察 (Observe)", "止损 (Cut Loss)"]


def decide_action(predicted_roas: float) -> Action:
    """根据预测 ROAS 给出投放建议。"""
    if predicted_roas > 5.0:
        return "放量 (Scale Up)"
    if predicted_roas >= 2.0:
        return "观察 (Observe)"
    return "止损 (Cut Loss)"


class PredictResponse(BaseModel):
    predicted_sales: str = Field(..., description="预测销售额, 严格保留两位小数的字符串")
    predicted_roas: str = Field(
        ...,
        description="预测 ROAS = 预测销售额 / 总花费, 严格保留两位小数的字符串",
    )
    action: Action = Field(..., description="基于 ROAS 阈值的投放建议")


class ModelState:
    """启动时加载并校验后的模型产物。"""

    def __init__(self, model, feature_names: list[str], categorical_dtypes: dict):
        self.model = model
        self.feature_names = feature_names
        self.categorical_dtypes = categorical_dtypes
        self.numeric_features = [
            name for name in feature_names if name not in categorical_dtypes
        ]


def preprocess(data: PredictRequest, state: ModelState) -> pd.DataFrame:
    """原始业务字段 -> 模型所需的 39 个特征 (列名/顺序严格来自 metadata)。

    说明: 新入参不包含历史花费, 因此 *_spend_lag_1 / total_spend_lag_1 /
    total_spend_roll_4_mean 采用"花费持续"近似(用当周花费代表近期花费)。
    sales_roll_8_mean 在未提供 8 周历史时为 NaN, LightGBM 可原生处理。
    """
    week_date = pd.Timestamp(data.date)
    iso_week = int(week_date.isocalendar().week)
    history = [float(v) for v in data.sales_history]
    hist4 = history[:4]

    spends = {col: float(getattr(data, col)) for col in SPEND_COLS}
    total_spend = float(sum(spends.values()))

    values = {
        # 日历 / 促销
        "holiday": int(data.holiday),
        "sales_promotion": data.sales_promotion,
        # 当周渠道花费及其派生
        **spends,
        "total_spend": total_spend,
        "n_active_channels": int(sum(v > 0 for v in spends.values())),
        # 时间特征
        "year": int(week_date.year),
        "month": int(week_date.month),
        "week": iso_week,
        "quarter": int(week_date.quarter),
        "dayofyear": int(week_date.dayofyear),
        "time_index": int((week_date - REFERENCE_MONDAY).days // 7),
        "week_sin": float(np.sin(2 * np.pi * iso_week / 52.18)),
        "week_cos": float(np.cos(2 * np.pi * iso_week / 52.18)),
        # 销售滞后: history[0] = t-1, ...[3] = t-4
        "sales_lag_1": history[0],
        "sales_lag_2": history[1],
        "sales_lag_3": history[2],
        "sales_lag_4": history[3],
        # 销售滚动统计 (与训练一致: 4 周窗口, 样本标准差 ddof=1)
        "sales_roll_4_mean": float(np.mean(hist4)),
        "sales_roll_4_std": float(pd.Series(hist4, dtype="float64").std(ddof=1)),
        "sales_roll_8_mean": (
            float(np.mean(history)) if len(history) >= 8 else float("nan")
        ),
        "sales_roll_4_max": float(max(hist4)),
        "sales_roll_4_min": float(min(hist4)),
        # 历史花费特征: 花费持续假设
        "total_spend_lag_1": total_spend,
        "total_spend_roll_4_mean": total_spend,
        **{f"{col}_lag_1": spends[col] for col in SPEND_COLS},
    }

    # 防御性校验: 生成的特征集合必须与模型/ metadata 完全一致
    missing = set(state.feature_names) - set(values)
    extra = set(values) - set(state.feature_names)
    if missing or extra:
        raise RuntimeError(
            f"preprocess 输出与模型特征不一致: 缺少={sorted(missing)}, 多余={sorted(extra)}"
        )

    # columns=feature_names 保证列名与顺序与训练时完全一致
    frame = pd.DataFrame([values], columns=state.feature_names)
    for col in state.numeric_features:
        frame[col] = frame[col].astype("float64")
    for col, dtype in state.categorical_dtypes.items():
        frame[col] = frame[col].astype(dtype)
    return frame


def _startup_check_payload() -> PredictRequest:
    """用于启动时验证预处理管道的最小合法请求。"""
    return PredictRequest(
        date=date_cls(2025, 1, 6),
        sales_promotion="Normal",
        holiday=0,
        competitor_spend=0.0,
        instagram_spend=0.0,
        google_ads_spend=0.0,
        tv_spend=0.0,
        youtube_spend=0.0,
        newspaper_spend=0.0,
        influencer_spend=0.0,
        ott_spend=0.0,
        sales_history=[1.0, 1.0, 1.0, 1.0],
    )


def load_model_state() -> ModelState:
    """加载模型与 metadata, 并做严格一致性校验; 任何不一致都让服务启动失败。"""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"模型文件不存在: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise RuntimeError(f"metadata 文件不存在: {METADATA_PATH}")

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(MODEL_PATH)  # 训练产物为 pickle 格式, joblib 可直接读取
    feature_names = list(metadata["features"])

    # 1) metadata 与模型内部特征名(含顺序)必须完全一致
    model_features = list(model.booster_.feature_name())
    if model_features != feature_names:
        raise RuntimeError(
            "metadata 特征顺序与模型不一致:\n"
            f"  metadata: {feature_names}\n"
            f"  model   : {model_features}"
        )
    if model.n_features_in_ != len(feature_names):
        raise RuntimeError(
            f"模型期望 {model.n_features_in_} 个特征, metadata 中为 {len(feature_names)} 个"
        )

    # 2) 类别特征的取值必须与训练时一致
    cat_features = list(metadata.get("categorical_features", []))
    pandas_categorical = getattr(model.booster_, "pandas_categorical", None) or []
    if len(cat_features) != len(pandas_categorical):
        raise RuntimeError("metadata 类别特征数量与模型内保存的类别信息数量不一致")

    categorical_dtypes = {
        name: pd.CategoricalDtype(categories=list(values), ordered=False)
        for name, values in zip(cat_features, pandas_categorical)
    }
    for name, dtype in categorical_dtypes.items():
        allowed = set(get_args(PredictRequest.model_fields[name].annotation))
        if allowed != set(dtype.categories):
            raise RuntimeError(
                f"{name} 的取值集合不一致: Pydantic={allowed}, model={set(dtype.categories)}"
            )

    state = ModelState(model, feature_names, categorical_dtypes)

    # 3) 启动时跑一遍预处理, 确保 39 个特征齐全且列顺序与 metadata 一致
    startup_frame = preprocess(_startup_check_payload(), state)
    if list(startup_frame.columns) != feature_names:
        raise RuntimeError("preprocess 输出的列顺序与 metadata 不一致")

    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_state = load_model_state()
    yield


app = FastAPI(
    title="Weekly Sales Forecast API",
    lifespan=lifespan,
    swagger_ui_parameters={
        "swagger_ui_css": "https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui.min.css",
        "swagger_ui_js": "https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui-bundle.js"
    }
)


def get_state(request: Request) -> ModelState:
    state = getattr(request.app.state, "model_state", None)
    if state is None:  # 理论上 lifespan 已保证模型加载
        raise HTTPException(status_code=503, detail="模型尚未加载")
    return state


@app.post("/predict", response_model=PredictResponse, summary="预测单周销售额")
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    state = get_state(request)

    total_spend = sum(float(getattr(payload, col)) for col in SPEND_COLS)
    if total_spend <= 0:
        raise HTTPException(
            status_code=400,
            detail="8 个渠道花费之和为 0，无法计算 ROAS",
        )

    try:
        row = preprocess(payload, state)
        prediction = float(state.model.predict(row)[0])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"预测失败: {exc}") from exc

    # 先格式化为严格两位小数的字符串, 再用同一个值做决策,
    # 避免展示值与 action 矛盾
    predicted_sales = f"{prediction:.2f}"
    predicted_roas = f"{prediction / total_spend:.2f}"
    return PredictResponse(
        predicted_sales=predicted_sales,
        predicted_roas=predicted_roas,
        action=decide_action(float(predicted_roas)),
    )


@app.get("/health", summary="健康检查")
def health(request: Request) -> dict:
    state = get_state(request)
    return {
        "status": "ok",
        "model_loaded": True,
        "model_file": MODEL_PATH.name,
        "n_features": len(state.feature_names),
    }
