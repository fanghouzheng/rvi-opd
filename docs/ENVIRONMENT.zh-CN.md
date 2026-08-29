# 环境与依赖

本仓库有两个必须隔离的运行环境。CPU 合同层用于配置校验、路由、统计与 synthetic smoke；确认性 GPU 环境用于 Relay/RvI 训练和 rollout。二者不能用一个宽松的 `torch>=...` requirements 合并。

## 1. 依赖文件

| 文件 | 用途 | 安装方式 |
|---|---|---|
| `requirements.txt` | CPU runtime | 无第三方包；本项目核心只用标准库 |
| `requirements-bootstrap.txt` | CPU 安装/构建工具 | 先安装；精确固定 pip 与 setuptools |
| `requirements-dev.txt` | CPU 开发与 CI 完整版本集合 | `pip install -r requirements-dev.txt` |
| `requirements-gpu-cu130.txt` | Relay 精确依赖锁的不可变入口 | 只供审计；实际安装使用 `scripts/create_gpu_env.sh` |
| `environment-cpu.yml` | 可选 Conda/Mamba CPU 基础环境 | `conda env create -f environment-cpu.yml` |
| `environment-lock.json` | 机器可读的 Python、平台、上游 commit 与关键包版本合同 | `scripts/check_environment.py` 读取 |

`pyproject.toml` 的 `dependencies=[]` 是刻意设计，不是漏写。`src/rvi_opd` 和当前 `unittest` 均不依赖 NumPy、PyTorch 或 Transformers。构建后端固定为 `setuptools==80.10.2`；开发文件同时固定 pytest/ruff 及其在 Python 3.9–3.12、Linux/macOS 上的完整间接依赖版本，不再由安装当天的解析结果决定。

## 2. CPU 开发环境

支持 Python `>=3.9,<3.13`；CI 固定覆盖 3.9 与 3.12。macOS、Linux 均可使用，不需要 CUDA。

### venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-bootstrap.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
make env-check
make validate
make test
make smoke
```

### Conda/Mamba

```bash
conda env create -f environment-cpu.yml
conda activate rvi-opd-cpu
python -m pip install -r requirements-bootstrap.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
make env-check
```

`make env-check` 会核对 Python 范围和开发包精确版本。CPU smoke 只证明软件合同可执行，不代表大模型训练成功。

## 3. 确认性 GPU 环境

唯一允许用于主表 head-to-head 的训练栈是：

| 项目 | 锁定值 |
|---|---|
| OS/架构 | Linux x86_64；上游验证于 Ubuntu 24.04 |
| Python | 3.12.13 |
| CUDA runtime/toolkit | 13.0 / 13.0.2 |
| pip / setuptools / wheel | 26.1.2 / 80.10.2 / 0.47.0 |
| PyTorch | 2.11.0 |
| vLLM | 0.21.0 |
| Transformers | 5.14.1 |
| datasets | 5.0.1 |
| Ray | 2.56.1 |
| Triton | 3.6.0 |
| NumPy | 1.26.4 |
| Relay-OPD | commit `eab21451f99e1a40fbb244f556de766d153c88f5` |

NVIDIA driver 属于宿主机条件，必须支持 CUDA 13.0，但不在 pip/Conda 中伪造一个跨集群固定值。每次正式 run 都要把 `nvidia-smi`、GPU 型号/拓扑、driver、CUDA、cuDNN、NCCL 和完整 `pip freeze` 写入 `runs/<run_id>/environment.json`。

### 自动创建

在一台具有 NVIDIA GPU、CUDA 13.0 兼容 driver 和 `python3.12` 的 Linux x86_64 机器上运行：

```bash
PYTHON_BIN=python3.12 ./scripts/create_gpu_env.sh
source .venv-gpu/bin/activate
python scripts/check_environment.py --profile gpu
rvi-opd validate-config --config-dir configs --run-ready
```

安装脚本只执行以下可审计流程：

1. 把 Relay clone 到被 `.gitignore` 排除的 `third_party/Relay-OPD`；
2. checkout 精确 commit；
3. 拒绝任何 tracked/untracked 改动，并校验其 `requirements.lock.txt` 的 SHA256 为 `693489b8ebb68350b9603fad07486c05e60fcd84aa2842305e19d2c6e26b5685`；
4. 调用上游 `create_locked_env.sh`，以 `--no-deps` 安装它验证过的完整 transitive lock；
5. 运行上游 vLLM patch、math grader 与 CUDA smoke；
6. 用 `--no-deps` 安装本项目，再核对 Relay editable source、commit/clean tree、完整 234 个 distribution pins、上游 strict verifier、PyTorch CUDA build 和 NVIDIA driver/GPU。

脚本会拒绝覆盖已有 `.venv-gpu`，也不会把已存在但 commit 不同的 Relay checkout 自动切换。需要自定义位置时使用：

```bash
RVI_RELAY_DIR=/data/upstreams/Relay-OPD \
RVI_GPU_VENV_DIR=/data/venvs/rvi-opd-gpu \
PYTHON_BIN=python3.12 \
./scripts/create_gpu_env.sh
```

### 为什么不能直接普通安装 GPU requirements

Relay 的锁定环境有一个已验证的 NumPy 1.26/OpenCV 4.13 组合，但 OpenCV metadata 会要求 NumPy 2。上游因此明确使用 `pip install --no-deps -r requirements.lock.txt`，随后执行自己的验证脚本。普通的 `pip install -r requirements-gpu-cu130.txt` 会重新解析依赖并可能改变该组合，禁止用于正式实验。

vLLM 也不能单独升级：Relay patch 依赖 vLLM 0.21.0 的私有 speculative-decoding 接口；PyTorch、CUDA、Triton、FlashInfer 与 vLLM wheel 又存在 ABI 耦合。

这里的“锁”指不可变上游文本、精确版本集合和 checkout 内容校验，不等同于所有平台 wheel 的字节级哈希锁。正式集群首跑还必须保存实际 wheelhouse/镜像 digest 与 `pip freeze`；若集群要求逐文件可复现，应基于目标 Linux/CUDA 平台生成带 SHA256 的内部 wheelhouse，不能把当前版本锁描述成容器级 bit-for-bit reproduction。

## 4. 禁止混装的上游

`upstreams.lock.json` 中的独立 verl commit `24f25b03aa4b54249a273655ebbcce06f484192b` 仅用于 reference/porting。它采用 vLLM 0.24、Transformers 5.5.3、NumPy 2 系列，与确认性 Relay 栈不兼容。

TRD fork 同样不得安装进 `.venv-gpu`。TRD 主表行应在 Relay 基栈中按冻结合同实现 clean-room adapter；若必须验跑原 fork，使用另一个环境，并把结果标为 stack-mismatched reproduction，不能直接进入 head-to-head。

以下命令在确认性环境中均禁止：

```bash
pip install -U torch transformers vllm numpy
pip install -e ".[hf]"
pip install -e /path/to/standalone-verl
pip install -e /path/to/trd
```

## 5. HealthBench evaluator

官方 grader 最好部署在独立 CPU/API 环境，使用锁定的 `simple-evals@652c89d0ca9df547706735883097e9537d40dc47`。不要为了 grader 的 `openai/blobfile/pandas` 依赖修改 GPU 训练环境。该 evaluator 环境必须在 W0 冻结实际包版本、grader template hash 与 `OPENAI_API_KEY` 的存在性；密钥只通过运行环境注入，禁止写入 requirements、配置、日志或 Git。

## 6. 运行前判定

环境成功不等于实验可开跑。正式 GPU run 必须同时通过：

```bash
python scripts/check_environment.py --profile gpu
rvi-opd validate-config --config-dir configs --run-ready
```

前者检查平台、Python、关键包、PyTorch CUDA build、GPU/driver；后者检查 C0/W0 必须冻结的 template、数据 split、rubric 与 coder hashes。任何一项失败都不得绕过。
