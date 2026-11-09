# PR #47361: Delete PagedAttention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-07-02 |
| 分类 | Attention / Kernel |
| Label | performance, ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 可用但注明 7 个文件因篇幅被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/47361 |

## 一句话结论

整体删除原版 PagedAttention CUDA 内核(v1/v2):推理默认路径已不用它,仅保留 ROCm 变体与基准脚本。

## 动机

vLLM 起家的自研 PagedAttention 内核(改编自 FasterTransformer 的 decoder masked multihead attention 模板)在 V1 推理中已不再被默认使用——attention 由 FlashAttention/FlashInfer 等后端承担,原内核成为纯维护负担:数百行模板 CUDA 代码、双版本(v1 无分区/v2 带分区)与配套 reduce 内核都要随每次 dtype/FP8/块稀疏特性更新。作者以一句 "It is time." 宣告删除时机成熟。

## 优化点

纯删除型改动(无新机制):移除 `csrc/libtorch_stable/attention/` 下的 `attention_kernels.cuh`(约 667 行,含 `paged_attention_kernel` 模板、`paged_attention_v1_kernel`、`paged_attention_v2_kernel` 与 `paged_attention_v2_reduce_kernel`)及 `paged_attention_v1.cu`/v2 相关 .cu 文件(pack 注明另有 7 个文件在 diff 中省略)。基准脚本 `benchmarks/kernels/benchmark_paged_attention.py` 改写为只针对 ROCm 的 `ops.paged_attention_rocm` 内核:删除 `--version v1/v2` 与 `--custom-paged-attn` 参数、v1/v2 分支与 CUDA 分区逻辑,启动时强制要求 ROCm 平台,head_size 选项收窄为 [64, 128],日志改为 "This script benchmarks the ROCm paged attention kernel. By default this is no longer used in vLLM inference."。性能意义是间接的:削减编译产物体积/编译时间与代码维护面,推理路径行为不变。

## 关键改动

```text
csrc/libtorch_stable/attention/attention_kernels.cuh: 删除(667 行 PagedAttention v1/v2 模板内核)
csrc/libtorch_stable/attention/paged_attention_v1.cu: 删除
benchmarks/kernels/benchmark_paged_attention.py: 改为仅基准 ROCm paged_attention_rocm
(其余约 7 个文件 pack diff 中省略,见 PR)
```

```python
logger.warning(
    "This script benchmarks the ROCm paged attention kernel. "
    "By default this is no longer used in vLLM inference."
)
if not current_platform.is_rocm():
    raise RuntimeError("This benchmark requires the ROCm platform.")
```

## 性能声明

PR 正文无任何性能数字(正文仅 "It is time.")。本 PR 不改变运行时推理路径,收益为工程性:删除死代码、减少 CUDA 编译量与维护成本。基准脚本保留只为继续度量仍在 ROCm 路径使用的 `paged_attention_rocm`。

## 验证与回归风险

PR 的 Test Plan/Test Result 章节为空,未列测试。风险在于编译面与引用完整性:若有外部代码或非默认配置仍引用 `ops.paged_attention_v1/v2`,将直接失效(pack diff 未见 Python 侧调用点删除,推测引用早已移除,PR 未注明);ROCm 的 `paged_attention_rocm` 与其 benchmark 保留不受影响。

## 关联

历史脉络:原 PagedAttention 为 vLLM 论文核心内核(对应早期 PR #1032 AWQ 同期的自研内核时代),被 FlashAttention/FlashInfer 等 backend 系列取代后删除。pack 素材未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
