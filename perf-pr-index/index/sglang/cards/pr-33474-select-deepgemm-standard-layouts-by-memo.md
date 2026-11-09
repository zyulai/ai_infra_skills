# PR #33474: Select DeepGEMM standard layouts by memory budget

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-08-05 |
| 分类 | Other (tier: major) |
| Label | quant, run-ci |
| Diff | 素材未提供 diffstat;diff 涉及 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/33474 |

## 一句话结论

把 DeepGEMM masked/compact 标准布局的选择从固定 expert 数阈值改为显存预算门控,masked 布局在可容纳时全面更快。

## 动机

PR 之前,标准 DeepGEMM dispatcher 仅凭 expert parallelism 决定是否用 masked 布局(`num_local_experts <= 32` 这类固定阈值)。但 masked workspace 随 local expert 数和 padded forward size 线性增长,固定阈值既会错过安全的性能机会,也可能选中超出可用显存的 shape 直接 OOM。动机来自 #33128 中关于布局选择的讨论。

## 优化点

新增 `SGLANG_DEEPGEMM_STANDARD_LAYOUT={auto|masked|compact}` 与 `SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION`(默认 0.25)。auto 模式下,从运行时 token 数、local expert 数、模型维度、激活 dtype 和 scale layout 估算 masked 峰值 workspace(`local_experts * padded_M * 三阶段中最大行字节数`),与预算比较决定是否用 masked。预算在 eager buffer 分配之后、CUDA Graph capture 之前,用跨 rank 最小空闲显存缓存一次;capture 路径不再查询空闲显存或同步(capture 中预算未设置时直接返回 False)。显式 `masked`/`compact` 覆盖仍然保留。默认 0.25 由 GB300 上 Qwen3.5-397B 的 OOM 边界标定:成功上限 20.1%、失败下限 37.6%,取中间保守值。

## 关键改动

```text
python/sglang/srt/environ.py: 新增 SGLANG_DEEPGEMM_STANDARD_LAYOUT 与 BUDGET_FRACTION 两个环境变量
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py: 核心选择逻辑(峰值估算 + 预算缓存 + auto 门控)
python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py: capture 前取跨 rank 最小可用显存设置预算
python/sglang/srt/layers/quantization/fp8.py: is_deepgemm_moe_runner_backend_enabled 支持传入 backend 以便 capture 期判断
test/registered/kernels/ops/moe/test_minimax_quant_scatter.py: 新增 auto 内存策略单测,旧 e2e 测试改为显式 layout 覆盖
```

```python
peak_row_bytes = max(
    input_row_bytes + input_scale_row_bytes + gateup_row_bytes,
    gateup_row_bytes + down_input_row_bytes + down_scale_row_bytes,
    down_input_row_bytes + down_scale_row_bytes + down_output_row_bytes,
)
return runner_config.num_local_experts * padded_m * peak_row_bytes
```

## 性能声明

均在 NVIDIA GB300、Qwen3.5-397B-A17B-FP8 shapes 上,masked 与 compact 逐对匹配(同拓扑/workload/backend/CUDA Graph 配置),共 50 组有效配对,masked 全胜:

- 隔离 standard runner(中位 eager 延迟):24/24 胜,equivalent throughput **+40.4% 到 +160.7%**(拓扑 TP4/EP1 至 TP8/EP8,local experts 64–512,M 512–16,384)。
- 完整 serving,ISL 8192 / OSL 1024(FP8 KV):12/12 胜,total TPS/GPU **+14.34% 到 +41.39%**(并发 64–512)。
- Prefill(OSL 1):14/14 胜,total TPS/GPU **+2.69% 到 +68.25%**。

显存标定:capture 前实测每卡空闲 170.48 GiB;最大成功 masked workspace 34.2 GiB(20.1%),最小 OOM 64.1 GiB(37.6%),默认 0.25 → 42.62 GiB。auto 策略 E2E 在 4x GB300 上 4 种 Captured M/拓扑组合全部零请求错误(如 8K/TP4-EP1 48/48、32K/TP4-EP2 192/192)。精度:底层路径在 #33128 已验证,本 PR 补跑 GSM8K 5-shot 1,314 题,**1,288 对 / 26 错,accuracy 0.9802130898**(16x GB300,PP2/TP8/EP1)。

## 验证与回归风险

新增 `test_standard_layout_auto_memory_policy` 单测覆盖预算门控边界;既有 masked vs compact 端到端一致性测试改为显式 layout 覆盖驱动。数值实现未改动(量化、scatter、grouped GEMM、activation、down projection、post-permute 均不变)。风险点:预算估算依赖模型维度、dtype、capture buckets 与 allocator 行为,默认 0.25 是刻意保守值而非通用显存保证,仍可通过环境变量调整;若 capture 期前预算未设置会静默回退 compact。pack 中记录的 CI 状态显示 Latest PR Test (Base) 与 (Extra) 均为失败(:x:)。

## 关联

前序 PR #33128(布局选择讨论来源,standard DeepGEMM 路径的精度验证)。无其他同区域 PR 编号记录。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
