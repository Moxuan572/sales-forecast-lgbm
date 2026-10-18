# 训练环境为 Python 3.14; 锁定依赖中的 numpy==2.5.4 要求 Python>=3.12,
# 因此这里选择满足全部锁定依赖的最低 slim 版本 (3.10/3.11 无法安装 requirements.txt)。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM 运行时依赖 OpenMP, 缺少 libgomp1 会报 libgomp.so.1 not found
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖, 充分利用 Docker 层缓存
# 构建时可通过 --build-arg PIP_INDEX_URL=... 切换 PyPI 镜像
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt

# 非 root 用户运行
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser main.py sales_forecast_lgbm.pkl sales_forecast_lgbm_metadata.json ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
