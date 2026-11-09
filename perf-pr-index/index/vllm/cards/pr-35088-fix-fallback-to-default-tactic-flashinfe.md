# PR #35088: Fix fallback to default tactic (flashinfer autotuner) with trtllm_fp4_block_scale_moe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @danisereb |
| 合并 | 2026-02-24 |
| 分类 | Quantization(major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 极小(1 个文件两处 flatten→reshape) |
| 链接 | https://github.com/vllm-project/vllm/pull/35088 |

## 一句话结论

修复 NVFP4 MoE 传给 flashinfer autotuner 的 hidden_states_scale 形状,使 trtllm_fp4_block_scale_moe 不再落到 default fallback tactic。

## 动机

flashinfer autotuner(见 `get_trtllm_moe_sm100_module`)按约定把 MoE 各张量第一维当作 num_tokens 参与 tuning bucket(`get_last_power_of_2_num_tokens_buckets(8192, 1)`)。vLLM 传 `hidden_states_scale` 时用 `.flatten()` 把张量压成一维,破坏该约定;开 `FLASHINFER_LOGGING_LEVEL=DEBUG` 可见 `Using fallback tactic for flashinfer::trtllm_fp4_block_scale_moe`,即 autotuner 未命中 tuned tactic,每次退回默认 tactic。

## 优化点

`flashinfer_trtllm_fp4_moe` 与 `flashinfer_trtllm_fp4_routed_moe` 两处把 `hidden_states_scale_linear_fp4.view(torch.float8_e4m3fn).flatten()` 改为 `.reshape(*hidden_states_fp4.shape[:-1], -1)`,保留 [num_tokens, ...] 前导维度,与 `Mxfp4MoEMethod.apply_monolithic` 中 `x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)` 的做法对齐。作者确认改后不再出现 "using fallback tactic" 打印。kernel 本身不变,变的是 autotuner 的 tuned tactic 能否被选中;这是让 tuning 生效的正确性修复,而非实测加速。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py: 两个入口函数的 hidden_states_scale 由 flatten 改为保留 token 维度的 reshape
```

```python
hidden_states_scale=hidden_states_scale_linear_fp4.view(
    torch.float8_e4m3fn
).reshape(*hidden_states_fp4.shape[:-1], -1),
```

## 性能声明

作者实测:"There was no change in `vllm bench serve` performance (tok/sec)",即其条件下 fallback tactic 与 tuned tactic 端到端吞吐相当,PR 未给出加速数字。精度回归:lm_eval gsm8k_llama(RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4,TP=2,max_model_len=4096,enable_chunked_prefill=True,enforce_eager,`VLLM_USE_FLASHINFER_MOE_FP4=1`、`VLLM_FLASHINFER_MOE_BACKEND=latency`),vLLM main 与本 PR 均 0.9371(flexible/strict exact_match,±0.0067),完全一致。

## 验证与回归风险

验证手段为作者手测的 lm_eval 精度对比与 DEBUG 日志检查;素材 diff 未新增单测。回归风险很小:连续张量下 reshape 与 flatten 元素总数一致,影响面是走 FLASHINFER_TRTLLM NVFP4 MoE backend 的两个入口;若 upstream flashinfer 对 scale 形状的假设变化,此处需同步。

## 关联

上游依赖 flashinfer autotuner 的 `get_trtllm_moe_sm100_module` 约定;同区域为 `Mxfp4MoEMethod.apply_monolithic`(本 PR 与其形状处理对齐)。素材未给出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
