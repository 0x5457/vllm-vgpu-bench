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

## 可选参数
- `--skip-baseline`
- `--python /path/to/python`
- `--hypervisor-bin /path/to/hypervisor`
- `--limiter-so /path/to/libcuda_limiter.so`
