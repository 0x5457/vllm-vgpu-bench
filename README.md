# vLLM vGPU Bench

单 GPU、低显存环境下，比较单进程与双进程 vLLM 的延迟/吞吐开销。

目标：观察 P99 与吞吐退化，不追求峰值吞吐。

## 快速开始
```bash
make install-python
make install-node

source .venv/bin/activate
make bench-matrix
```

## 配置要点
- 单 GPU（`CUDA_VISIBLE_DEVICES` 仅一个）
- 模型：`Qwen/Qwen2.5-0.5B`，FP16，`--max-model-len 2048`
- 显存：单进程 `0.75`，双进程 `0.36`（8 GiB 默认）

## 常用命令
```bash
make start-single
make start-double
make bench-matrix
```

## 限流矩阵（hypervisor local）
```bash
source .venv/bin/activate
npx tsx scripts/run_matrix.ts \
  --modes single,double \
  --utils 0.3,0.5,0.7
```

输出在 `results/YYYYMMDD_HHMMSS/`，包含 `summary.csv` / `summary.json` 以及日志。
`bench-matrix` 会自动清理 vLLM/限流子进程并在每轮之间冷却。

## 可选参数
- `--skip-baseline`
- `--python /path/to/python`
- `--hypervisor-bin /path/to/hypervisor`
- `--limiter-so /path/to/libcuda_limiter.so`

## 环境变量
- `VLLM_MAX_MODEL_LEN`：统一设置 max model len
- `VLLM_GPU_UTIL_SINGLE`：单进程显存占用比例
- `VLLM_GPU_UTIL_DOUBLE`：双进程显存占用比例
- `COOLDOWN_S`：`bench-matrix` 每轮结束后的冷却秒数（默认 5）
- `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE`：模型缓存目录（默认：项目内 `.cache/huggingface`）

## 指标采集
- `bench-matrix` 会用 `nvidia-smi` 采集 GPU 指标（1s 采样），落盘到每个 run 的 `summary.json` 中。
- `summary.csv` 追加字段：
  - `gpuSamples`
  - `gpuAvgUtilGpu`, `gpuAvgUtilMem`
  - `gpuAvgMemUsedMiB`, `gpuAvgMemTotalMiB`
  - `gpuAvgPowerW`, `gpuAvgTempC`
  - `gpuMaxUtilGpu`, `gpuMaxUtilMem`
  - `gpuMaxMemUsedMiB`, `gpuMaxMemTotalMiB`
  - `gpuMaxPowerW`, `gpuMaxTempC`
