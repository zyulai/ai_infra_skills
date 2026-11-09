# PR #53065: [XPU][MoE] Tune Triton fused MoE for Intel XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pmanczak |
| 合并 | 2026-10-07 |
| 分类 | MoE（tier: major） |
| Label | performance, intel-gpu, ready |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 `benchmarks/kernels/benchmark_moe.py` 并新增 3 个 Intel Arc Pro B70 调优 JSON 配置（`E=128,N=384`、`E=64,N=1024`、`E=64,N=512`） |
| 链接 | https://github.com/vllm-project/vllm/pull/53065 |

## 一句话结论

Intel XPU 的 Triton fused MoE 此前无调优配置、一直走默认 heuristic；本 PR 打通 XPU 调优流程并加入 Arc Pro B70 配置，kernel 几何均值提速 1.86–1.92x。

## 动机

Triton fused MoE kernel 可以在 Intel XPU 上运行，但 `vllm/model_executor/layers/fused_moe/configs/` 中没有任何 XPU 设备的调优 JSON，运行时总是回退到 `get_default_config()` 的通用 heuristic，B70 这类消费级 Arc 卡的 MoE GEMM 效率明显低于可达水平。需要让 `benchmark_moe.py` 的自动调优流程支持 XPU，并产出设备/形状专属配置。

## 优化点

三块改动。其一，`benchmark_moe.py` 新增 `get_xpu_tuning_space`（BLOCK_M 16–256、BLOCK_N 64/128、BLOCK_K 32/64、num_warps 4/8/16、GROUP_M 1/8/16、num_stages 3/4、grf_mode 固定 "256"——实测 grf_mode=256 全面胜过 128，BLOCK_N=32 与 num_warps=32 从未胜出），并加 `prune_xpu_search_space`：按"每 expert 实际分到的行数"（`rows_per_expert = ceil(num_tokens*topk/num_experts)`）裁剪 BLOCK_SIZE_M，避免宽 tile 只会 padding 且编译最慢。其二，`sort_config` 序列化时保留 `grf_mode` 字段，否则配置加载后会丢失。其三，去除仅有的两个 CUDA 专属调用阻塞：worker 默认设备改用 `current_platform.device_type`，图捕获改为按平台选 `torch.xpu.XPUGraph` 并经 `current_platform.graph()`（与 `benchmark_selective_state_update.py` 一致）。Ray 路径不变；XPU 上需装 dpctl，调优应单卡运行（`ZE_AFFINITY_MASK=<id>`），多卡可见时 Ray 设置的 per-worker affinity 会在 Level Zero 初始化之后到达，全部 worker 落到 device 0。

## 关键改动

```text
benchmarks/kernels/benchmark_moe.py: XPU 调优搜索空间与按 rows_per_expert 裁剪；XPUGraph/default device 适配；序列化保留 grf_mode
configs/E=64,N=1024,...B70.json: OLMoE tp=1 形状，18 个 batch 档位配置
configs/E=64,N=512,...B70.json: OLMoE tp=2 形状，6 个 batch 档位配置
configs/E=128,N=384,...B70.json: Qwen3-30B-A3B/Qwen3-Coder tp=2 形状，18 个 batch 档位配置
```

```python
rows_per_expert = (num_tokens * topk + num_experts - 1) // num_experts
max_block_m = max(16, triton.next_power_of_2(rows_per_expert))
return [c for c in search_space if c["BLOCK_SIZE_M"] <= max_block_m]
```

## 性能声明

硬件 Intel Arc Pro B70，bf16；baseline 为 Triton 内置 heuristic，tuned 为生成的 JSON。Kernel 级（18 个 batch 档位全扫，几何均值）：1.86x（E=64,N=1024）、1.91x（E=64,N=512）、1.92x（E=60,N=1408，验证形状）；逐 batch 峰值约 3.0–3.4x（batch 128–2048 段），如 OLMoE E=64,N=1024 batch 128 为 3.02x、Qwen1.5-MoE batch 2048 为 3.36x。E2E（`--moe-backend triton`，512 输入/128 输出 token、32 prompts）：OLMoE TP1 3022.8→4146.4 tok/s（1.37x）、OLMoE TP2 1.13x、Qwen1.5-MoE 1.070x（12 样本/臂，p=7.4e-07）、DeepSeek-V2-Lite TP2 1.023x（4 样本/臂，p=0.0022）、Qwen3-Coder TP4 1.041x（+4.10% mean，95% bootstrap CI [+3.09%, +5.88%]）。另：Qwen3-Coder tp=4 本地形状 E=128,N=192（未随本 revision 发布）18 个调优配置通过 fp32 验证（rel_l2=3.82e-3..4.20e-3），microkernel 延迟较 heuristic 降 43.73% mean / 43.34% median（仅聚合数字）。

## 验证与回归风险

测试流程：每个生成的优胜配置对照 fp32 参考验证（rel_l2 约 4.1e-3 量级）；确认 serving 以 `--moe-backend triton` 加载确切 JSON；ruff check/format 通过。CUDA/ROCm 保持 `torch.cuda.CUDAGraph` 与 `torch.cuda.graph` 不变，且本 PR 未在 CUDA/ROCm 上运行。风险面：影响面集中在 XPU 平台的 benchmark 脚本与新增配置文件，CUDA/ROCm 路径仅经 `is_cuda_alike()` 分支保护；E=64,N=704（DeepSeek-V2-Lite）等验证形状未发布配置，收益需待后续提交；XPU 多卡 Ray 调优存在 affinity 时序限制（须单卡跑）。

## 关联

同分类（fused MoE 调优配置）索引内另有：#28373（AMD Triton MoE tuning）、#62763（H200 FP8 fused MoE config）。PR 正文未列其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
