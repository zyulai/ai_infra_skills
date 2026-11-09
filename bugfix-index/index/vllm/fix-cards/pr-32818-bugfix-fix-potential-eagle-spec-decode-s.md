# PR #32818: [Bugfix] Fix potential EAGLE spec decode segfault during graph capture

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32818](https://github.com/vllm-project/vllm/pull/32818) |
| 作者 | mawong-amd |
| 合并日期 | 2026-01-22 |
| labels | bug, ready, speculative-decoding, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
重构后 dummy_run 丢失 use_cudagraphs 判断导致不该走 CUDA graph 时也 dispatch,MI300X EAGLE TP>1 捕获期段错误;恢复原逻辑修复。

## 问题现象
MI300X 上,EAGLE spec decode 且 TP > 1 时,CUDA graph 捕获过程中出现段错误(segfault)。触发方式例如:
`vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 2048 -tp 2 --max-num-batched-tokens 2048 --speculative-config='{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 3, "max_model_len": 2048}'`
或 Llama-4-Scout + `--load-format dummy --max-num-seqs 1 --compilation-config='{"cudagraph_capture_sizes": [1]}'` 的组合。此外还导致 `tests/v1/e2e/test_spec_decode.py::test_eagle_correctness` 等测试失败。

## 根因
正文披露:PR #30143 重构后,`SpecDecodeBaseProposer.dummy_run` 中 `use_cudagraphs` 参数被遗留为未使用,导致原本不应发生的 CUDA graph dispatch 错误发生,在 MI300X 上表现为 EAGLE spec decode TP>1 时 CUDA graph 捕获期间的段错误。本 PR 恢复 `dummy_run` 的原始逻辑(调查与修复亦归功于 @micah-wil)。

## 修复方案
修改 `vllm/v1/spec_decode/eagle.py` 的 `dummy_run`:恢复对 `use_cudagraphs` 参数的判断——为 True 时才调用 `cudagraph_dispatcher.dispatch(num_tokens_dp_padded)` 获取 batch_desc;否则 `cudagraph_runtime_mode = CUDAGraphMode.NONE`,`num_input_tokens = num_tokens_dp_padded`。

关键 diff 摘录:
```python
if use_cudagraphs:
    cudagraph_runtime_mode, batch_desc = (
        self.cudagraph_dispatcher.dispatch(num_tokens_dp_padded)
    )
    num_input_tokens = batch_desc.num_tokens
else:
    cudagraph_runtime_mode = CUDAGraphMode.NONE
    num_input_tokens = num_tokens_dp_padded
```

## 影响与回归风险
- 回归面:改动集中在 `SpecDecodeBaseProposer.dummy_run` 的 CUDA graph dispatch 分支,影响所有 spec decode proposer 的 dummy_run 路径;恢复重构前的行为,主要作用于 DP padding 后的 graph 模式选择。

## 关联
- [PR #30143](https://github.com/vllm-project/vllm/pull/30143)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
