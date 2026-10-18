# 销售预测服务（LightGBM + FastAPI + Docker + Nginx）

基于 Marketing Mix Modeling（MMM）周度数据训练 LightGBM 回归模型，预测下一周销售额。
调用方只需提供原始业务字段，服务端自动生成 39 个模型特征。项目已容器化，并通过 Nginx 反向代理部署到 AWS EC2。

## 架构

```
Client
  │  HTTP :80
  ▼
Nginx（sales-forecast-nginx）
  │  proxy_pass http://api:8000
  ▼
FastAPI（sales-forecast-api）
  │  joblib 加载
  ▼
sales_forecast_lgbm.pkl + sales_forecast_lgbm_metadata.json
```

只有 Nginx 对外暴露 80 端口，FastAPI 的 8000 端口仅在 Docker 网络内可见。

## 目录结构

```
.
├── train_model.py                        # 训练、评估、导出模型
├── mmm_dataset.csv                       # 原始周度数据（209 行）
├── sales_forecast_lgbm.pkl               # 训练好的 LightGBM 模型
├── sales_forecast_lgbm_metadata.json     # 特征顺序、类别取值、指标
├── main.py                               # FastAPI 服务 + preprocess()
├── sample_payload.json                   # 请求示例
├── requirements.txt                      # 锁定依赖版本
├── Dockerfile                            # API 镜像
├── .dockerignore
├── nginx.conf                            # 反向代理配置
├── docker-compose.yml                    # api + nginx 编排
└── README.md
```

## 数据与特征

- 数据范围：2022-01-03 至 2025-12-29，共 209 个自然周。
- 缺失值处理：
  - 8 个渠道花费列共 434 个缺失单元格 → 填 `0`，表示该周该渠道无花费记录。
  - `sales` 目标列缺失 3 行 → 不填充，建完特征后剔除，最终使用 206 行。
  - 滞后/滚动特征开头产生的 NaN → 保留，由 LightGBM 原生处理。
- 39 个特征：

| 分组 | 特征 |
|---|---|
| 时间（8） | `year` `month` `week` `quarter` `dayofyear` `time_index` `week_sin` `week_cos` |
| 业务与花费（12） | `holiday` `sales_promotion` 8 个渠道花费 `total_spend` `n_active_channels` |
| 销售滞后（4） | `sales_lag_1` ~ `sales_lag_4` |
| 销售滚动（5） | `sales_roll_4_mean/std/max/min` `sales_roll_8_mean` |
| 花费滞后/滚动（10） | 8 个渠道 `*_lag_1` `total_spend_lag_1` `total_spend_roll_4_mean` |

防泄漏：所有 lag/rolling 特征均基于 `shift(1)+`，按时间顺序切分，不打乱数据。

### 模型指标

| 指标 | 留出集（最后 42 周） | 4 折时间序列 CV |
|---|---:|---:|
| MAE | **127,477** | 206,181 |
| R² | **0.859** | 0.405 |
| RMSE | 155,671 | — |
| MAPE | 6.42% | — |

> 留出集指标来自按时间切分训练出的模型；导出的 `.pkl` 随后在全部 206 行上重新拟合。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查，返回模型是否加载成功 |
| `POST` | `/predict` | 预测单周销售额 |
| `GET` | `/docs` | Swagger UI |

请求字段：

- `date`：待预测周日期（ISO 格式，不早于 `2022-01-03`）
- `sales_promotion`：`BOGO` / `Coupons` / `Flash Sale` / `Normal`
- `holiday`：`0` 或 `1`
- 8 个渠道花费：非负浮点数
- `sales_history`：过去 4–8 周销售额，按 **近 → 远** 排列，如 `[本周, 上周, 上上周, 上上上周]`

示例：

```bash
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

```json
{"predicted_sales": 2434879.816363135}
```

预处理说明：

- `sales_lag_1~4`、`sales_roll_4_*` 由 `sales_history` 的前 4 个值精确计算。
- 只传 4 个历史值时，`sales_roll_8_mean` 为 NaN（模型可原生处理）；传 8 个可精确计算。
- 入参不包含历史花费，`*_spend_lag_1`、`total_spend_lag_1`、`total_spend_roll_4_mean` 
  使用“花费持续”假设，以当周花费近似上周花费。
  这一简化会使留出集误差从 MAE 127,477 / R² 0.859 降到约 150,612 / 0.782。

## 本地运行

需要 Python 3.12+（`numpy==2.5.4` 的最低要求）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

验证：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d @sample_payload.json
```

## Docker

```bash
docker build -t sales-forecast-api:latest .
docker run -d --name sales-forecast-api -p 8000:8000 sales-forecast-api:latest
```

国内网络访问 PyPI 较慢时，构建时切换镜像源：

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t sales-forecast-api:latest .
```

镜像基于 `python:3.12-slim`，并安装了 LightGBM 运行所需的 `libgomp1`。

## Docker Compose + Nginx

确保本地已有 `sales-forecast-api:latest` 镜像（见上一节），然后：

```bash
docker compose up -d
docker compose ps
```

验证：

```bash
curl http://localhost/health
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" -d @sample_payload.json
```

Compose 启动两个服务：

- `api`：`sales-forecast-api:latest`，仅 `expose 8000`，不映射到宿主机。
- `nginx`：`nginx:alpine`，宿主机 `80 → 容器 80`，挂载 `nginx.conf`，等待 API 健康后启动。

常用命令：

```bash
docker compose logs -f          # 查看日志
docker compose restart          # 重启
docker compose down             # 停止并删除容器
```

## 部署到 EC2

### 1. 安全组

入站规则只开放：

| 端口 | 来源 | 用途 |
|---|---|---|
| 22 | 你的公网 IP | SSH |
| 80 | 0.0.0.0/0 | HTTP（Nginx） |

不需要开放 8000。

### 2. 安装 Docker 与 Compose

以 Ubuntu 为例：

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # 重新登录后生效
```

确认：

```bash
docker --version
docker compose version
```

Amazon Linux 使用 `sudo dnf install -y docker` 并启动 `docker` 服务，Compose 需另行安装 v2 插件。

### 3. 上传项目

在本地项目目录执行（Ubuntu 实例用户为 `ubuntu`，Amazon Linux 为 `ec2-user`）：

```bash
rsync -av --exclude '.venv' --exclude '__pycache__' \
  -e "ssh -i ~/your-key.pem" ./ \
  ubuntu@<EC2_PUBLIC_IP>:~/sales-forecast/
```

也可以把项目推到 Git 仓库，再在 EC2 上 `git clone`。

### 4. 构建并启动

```bash
cd ~/sales-forecast

docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t sales-forecast-api:latest .

docker compose up -d
docker compose ps
```

### 5. 从本机验证

```bash
curl http://<EC2_PUBLIC_IP>/health

curl -X POST http://<EC2_PUBLIC_IP>/predict \
  -H "Content-Type: application/json" -d @sample_payload.json
```

浏览器打开 `http://<EC2_PUBLIC_IP>/docs` 可查看 Swagger UI。

### 6. 更新版本

代码或模型更新后，重新上传文件，然后在 EC2 上：

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t sales-forecast-api:latest .

docker compose up -d --force-recreate
```

> `docker-compose.yml` 中使用的是本地镜像 `sales-forecast-api:latest`，没有 `build:` 配置，
> 所以需要先 `docker build`，再 `docker compose up`。

## 常见问题

| 问题 | 处理 |
|---|---|
| `libgomp.so.1: cannot open shared object file` | 镜像中已安装 `libgomp1`；自建镜像时不要漏掉 |
| Python 3.10/3.11 安装依赖失败 | `numpy==2.5.4` 要求 Python ≥ 3.12，请使用 `python:3.12-slim` |
| 容器内 `pip install` 超时 | 构建时加 `--build-arg PIP_INDEX_URL=...` |
| Docker Hub 拉取超时 | 使用镜像加速地址拉取后重新打 tag，例如 `docker.m.daocloud.io/library/python:3.12-slim` |
| 80 端口被占用 | 修改 `docker-compose.yml` 为 `"8080:80"`，改用 `http://host:8080` 访问 |
| 启动时报特征顺序不一致 | `.pkl` 与 `metadata.json` 必须来自同一次训练，不能混用 |
| 公网 IP 每次重启变化 | 为实例绑定 Elastic IP |

生产环境建议在 Nginx 前增加 HTTPS（443）与证书，或使用 ALB + ACM。
