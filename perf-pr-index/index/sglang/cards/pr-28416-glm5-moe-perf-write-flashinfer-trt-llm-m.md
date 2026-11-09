# PR #28416: [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mattteochen |
| 合并 | 2026-07-20 |
| 分类 | MoE (tier: major) |
| Label | deepseek, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/28416 |

## 一句话结论

利用 FlashInfer 0.6.14 的 output 参数让 TRT-LLM FP8 block-scale MoE kernel 直接写入 symmetric memory 输出缓冲,删掉 kernel 后的 `copy_` 与双流 copy/add workaround,GLM5-FP8 上 1k1k Pareto 全并发稳定 +1%。

## 动机

SGLang 把 FlashInfer MoE 结果分配在 symmetric memory 中,但旧版 FlashInfer 的 monolithic `trtllm_fp8_block_scale_moe` API 不支持写入该缓冲,结果必须额外做一次 `copy_`;本 PR 的早期版本曾用双流 copy/add workaround 来隐藏这次拷贝。SGLang 现已 pin FlashInfer 0.6.14,该版本为 non-routed 与 routed 两个 FP8 block-scale MoE API 都暴露了 `output` 张量,两个 kernel 可以直接写 symmetric 输出缓冲,调度 workaround 与拷贝都成为多余。

## 优化点

机制上是消除 MoE 输出的一次全量 HBM 拷贝:新增 mutation-only 的 SGLang custom-op wrapper(`trtllm_fp8_block_scale_moe_out_wrapper` 与 `trtllm_fp8_block_scale_routed_moe_out_wrapper`),把 `output` 作为被变更参数传入(`mutates_args=["output"]`,返回 None);特意与原有"分配返回"型 wrapper 分开,避免 custom-op 返回的 tensor 与被变更参数产生别名。调用侧 `fused_experts_none_to_flashinfer_trtllm_fp8` 把 `symm_output` 直接传给两个 FlashInfer kernel,删除 kernel 之后的 `symm_output.copy_(output)`(及其 TODO 注释)与 runtime-context/shared-expert 的双流 copy/add workaround。同时新增针对两条 output-buffer 路径的真实 FlashInfer 回归测试。

## 关键改动

```text
python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py: 新增 *_out_wrapper(mutates_args=["output"]),透传 output 给 FlashInfer API
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: 传 symm_output 给两个 kernel,删除 symm_output.copy_(output)
```

关键片段(调用侧改动):

```python
trtllm_fp8_block_scale_routed_moe_out_wrapper(
    ..., output=symm_output, ...)
# removed: symm_output.copy_(output)
output = symm_output
```

## 性能声明

环境:8xB200,GLM5-FP8:

- 单 batch 基准(B=4):median throughput 302.53 -> 306.24 token/s
- 1k1k Pareto(isl/osl 各 1k,conc 4-256):每个并发点稳定 1% speedup;如 conc=4 total TPS/GPU 75.44 -> 76.46(1.01x),conc=256 1111.88 -> 1127.13(1.01x);TTFT 基本持平(约 1.00x)
- profiling 显示 `memcopy_post` 消失(PR 附截图)

准确性:repeat 6 次 mean 0.863(scores 0.843-0.884),mean latency 14595.108 s(PR 未注明该评估的数据集与命令);应作者要求未跑 GLM 模型测试(避免权重下载)。

## 验证与回归风险

FlashInfer 0.6.14 验证(B200,PyTorch 2.11.0+cu130):non-routed DeepSeekV3 grouped-routing API 返回所供 output storage 且与 allocating 路径完全一致(`max_abs_diff=0`);新增 `test/registered/unit/layers/moe/test_flashinfer_trtllm_moe_output.py` 2 passed。回归面:仅 FlashInfer TRT-LLM FP8 block-scale MoE(routed 与 non-routed 两条 output 路径);wrapper 签名从"返回 tensor"变为"变更 output",所有调用方需同步更新(diff 已覆盖调用点);依赖 FlashInfer >= 0.6.14 的 `output` 参数能力;确定性推理与 CUDA graph 路径需保证 symm_output 缓冲生命周期;CI Base/Extra 为失败标记。

## 关联

上游依赖:flashinfer-ai/flashinfer v0.6.14 的 `output` 参数与 issue #2703(旧 copy 的 TODO 所指,现已通过本 PR 移除)。同区域为 FlashInfer TRT-LLM MoE 输出路径优化;pack 未列出同分类其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
