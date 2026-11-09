# PR #28879: [Feature] Shared Experts Overlap with FI deepgemm swap kernel, 2.2% throughput improvement and 3.6% TTFT improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-11-21 |
| 分类 | MoE (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/28879 |

## 一句话结论

在 modular MoE kernel 路径上启用 shared experts 独立 CUDA stream 重叠执行,DSv3.1 TP8 BS32 吞吐 +2.2%、TTFT -3.6%。

## 动机

修复 PR #27134 评论中反馈的问题:modular kernel(`FusedMoEKernel`)路径上的 shared experts 与 fused MoE 主 kernel 串行执行,而 layer 级(`forward_impl`)已有 stream 重叠逻辑。shared experts 本可与 routed experts 的 deepgemm/swap kernel 并行,但在 modular 路径中 `shared_experts_stream` 未被传入,重叠不生效,小 batch(BS32)下这段串行开销直接体现在 TTFT 上。

## 优化点

把 layer 的 `shared_experts_stream` 通过 `fused_moe_modular_method.make` 传入 `FusedMoEKernel`,并在 modular kernel 内部实现与 layer 级等价的流重叠逻辑:(1) `_maybe_setup_shared_experts_stream` 在主 stream 上先 clone hidden_states(避免 routed expert kernel 原地修改导致竞争),`record_stream` 防止 clone 被提前回收,然后 `shared_experts_stream.wait_stream(current_stream())` 设定同步起点,使 shared experts 可与后续 router/gate 及主 MoE kernel 并行;(2) `_finalize` 中 `maybe_run_shared_experts` 决定在 dedicated stream 上跑 `self.shared_experts(hidden_states_clone)` 还是回落主 stream;(3) `_wait_for_shared_experts_stream` 在 shared_output 被消费前 `current_stream().wait_stream(shared_experts_stream)` 保证完成序。另将 `prepare_finalize.py` 中 `apply_router_weight_on_input` 的 `a1.mul_(...)` 改为非原地 `a1 = a1 * ...`,避免与 stream 重叠路径的原地冲突。超过 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`(256)时不启用。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/modular_kernel.py: FusedMoEKernel 接收 shared_experts_stream;新增 stream setup/wait 逻辑,finalize 中 shared experts 可在独立 stream 运行
vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py: make 时把 moe_layer.shared_experts_stream 传入 kernel
vllm/model_executor/layers/fused_moe/layer.py: 原 forward_impl 内联重叠逻辑抽取为 _maybe_setup_shared_experts_stream
vllm/model_executor/layers/fused_moe/prepare_finalize.py: apply_router_weight_on_input 的 mul_ 改为非原地
```

关键代码摘录(节选,<=15 行):

```python
def maybe_run_shared_experts() -> torch.Tensor | None:
    if self.shared_experts is None:
        return None
    if (not use_shared_experts_stream or ...):
        # fall back to running on the current stream
        return self.shared_experts(hidden_states)
    assert hidden_states_clone is not None
    # launch shared experts on the dedicated stream.
    with torch.cuda.stream(self.shared_experts_stream):
        return self.shared_experts(hidden_states_clone)
```

## 性能声明

- 条件:DeepSeek-V3.1,`-tp 8 --enable-expert-parallel`,vllm bench serve(random-input-len 2 / random-output-len 100 / 256 prompts / ignore-eos):
  - batch size 32(max-concurrency 32):Request throughput 11.25 → 11.50 req/s(+2.2%);Mean TTFT 53.12 → 51.22 ms(-3.6%);Mean TPOT 28.16 → 27.54 ms。
  - batch size 256(max-concurrency 256):throughput 31.35 → 31.63 req/s,Mean TTFT 2178.18 → 2166.35 ms,改善幅度较小但仍优于无重叠。
  - 注:batch size > 256 时因 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` 不启用。
- 准确性:lm_eval gsm8k(5-shot,num_concurrent 1024):flexible-extract / strict-match exact_match 均 0.9591(±0.0055)。

## 验证与回归风险

- 验证:gsm8k 精度对照 + vllm bench serve A/B;perfetto trace(正文图片)显示 shared experts 与主 MoE kernel 时间轴重叠。pack 未列新增单测。
- 回归风险面:modular kernel 的所有 shared-experts 模型(DeepSeek 系)在 CUDA graph capture 下多 stream 的正确性(clone 必要性注释即为此);`record_stream`/`wait_stream` 顺序错误会造成竞态;非 CUDA 平台与 token 超阈值时回落主 stream;`a1` 非原地化的显存与量化路径行为需与各 quant method 兼容。

## 关联

- 前序/背景:PR #27134(相关评论触发本修复);同区域:索引库已有 pr-28164(TP/EP shared experts overlap)、pr-44313(ROCm fused shared expert)、pr-61457(per-layer fused shared experts)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
