# PR #39822: [Hybrid] Warmup Mamba2 SSD kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tdoublep |
| 合并 | 2026-05-12 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 3 个文件（mamba_mixer2.py、gdn_linear_attn.py、config/model.py，约 +110/-8 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/39822 |

## 一句话结论

把 Mamba2 SSD Triton kernel 的 auto-tune 从首个推理请求提前到 V1 profile 阶段执行，冷缓存首请求延迟从 31.3 s 降到 2.9 s（−91%）。

## 动机

Mamba2 SSD kernel 的 Triton auto-tuner 采用惰性触发：第一个推理请求到来时才 sweep 配置，造成首请求出现约 31 s 的巨大延迟尖峰。`TRITON_PRINT_AUTOTUNING=1` 显示 main 上 58 个 kernel autotuning 事件发生在模型加载之后的首次请求中。应把这一代价转移到服务器启动阶段。

## 优化点

在 `MambaMixer2` 新增 `_warmup_ssd_kernels()`：V1 profile run（`attn_metadata is None` 分支，SSM cache 分配之前、显存尚充裕时）用 dummy tensor 跑一次最小 `mamba_chunk_scan_combined_varlen` 前向，触发 Triton auto-tuning。要点：(1) 每个 Mamba2 层只 warmup 一次（`_ssd_kernels_warmed_up` 标志），后续层直接命中 Triton cache；(2) autotune cache key 依赖 dtype 与 head 维度而与 seqlen/batch 无关，故用单一形状（seqlen=chunk_size、batch=1）即可覆盖；(3) 显式使用真实推理的 dtype（含 `get_state_dtype()` 的 `ssm_state_dtype`）以匹配 cache key；(4) `_state_passing_fwd` 与 `_chunk_scan_fwd` 以 `HAS_INITSTATES` 为 constexpr 产出两套编译产物，warmup 循环 `use_initial_states in (False, True)` 把两条路径都覆盖；(5) warmup 失败时降级为 warning（首请求仍可能出现延迟尖峰/OOM），随后 `torch.accelerator.empty_cache()`。附带改动：`get_mamba_chunk_size()` 返回类型从 `int | None` 收紧为 `int`，Mamba1 的默认 chunk size 注释由 1024 更正为 2048；`gdn_linear_attn.py` 的 `_prefill_kernels_warmed_up` 标志初始化方式顺带修正（`hasattr` 检查改为实例标志）。

## 关键改动

```text
vllm/model_executor/layers/mamba/mamba_mixer2.py: 新增 _warmup_ssd_kernels（profile run 触发 SSD kernel autotune，覆盖 HAS_INITSTATES 两条路径），forward 的 profile 分支调用
vllm/model_executor/layers/mamba/gdn_linear_attn.py: _prefill_kernels_warmed_up 改为显式实例标志
vllm/config/model.py: get_mamba_chunk_size 返回类型收紧为 int
```

触发点：

```python
if attn_metadata is None:
    # V1 profile run -- warm up SSD kernels so that autotuning
    # completes before SSM cache allocation.
    self._warmup_ssd_kernels(projected_states)
```

## 性能声明

测试条件：单张 H100 80GB、`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`（Mamba2 hybrid）、`max_model_len=512`、冷 Triton cache（清空 `~/.triton/cache` 与 torch compile cache）。结果：首请求延迟 31.343 s → 2.890 s（**−28.5 s，−91%**）；模型加载时间 30.0 s → 76.9 s（+46.9 s，autotuning 成本移到此处）；后续请求均 0.083 s 不变；首请求/后续请求比从 378x 收窄到 35x（作者标注 **10.8x improvement**）。autotuning 事件：main 在模型加载后发生 58 个，warmup 后为 0。残留约 2.9 s 首请求开销（相对 0.08 s）不属于 autotuning（期间无 autotune 输出），作者怀疑来自 Triton JIT 编译，留作后续调查。

## 验证与回归风险

Test plan 前三项已勾选（autotuning 位置确认、首请求延迟确认、后续请求不变确认），"Run existing Mamba2 model tests" 未勾选——素材未显示该测试已跑。风险面：(1) warmup 代价全部转移到启动：模型加载 +46.9 s，对频繁重启的服务有影响；(2) warmup 用 try/except 兜底，失败只 warning，线上仍可能首请求尖峰；(3) warmup 形状假设 autotune key 与 seqlen/batch 无关，若 Triton 版本改变 key 组成可能漏覆盖；(4) dtype 匹配依赖 `get_state_dtype()` 正确反映真实推理 dtype。

## 关联

同作者与同区域：PR #14152 亦为 @tdoublep 的 Triton backend 性能优化（不同对象）。Mamba2/GDN hybrid 线的其他优化素材未列出编号；`gdn_linear_attn.py` 中已存在的 `_warmup_prefill_kernels` 是同类模式（GDN prefill kernel 预热），本卡将其模式推广到 Mamba2 SSD。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
