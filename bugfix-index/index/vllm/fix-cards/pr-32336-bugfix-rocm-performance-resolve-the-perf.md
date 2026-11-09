# PR #32336: [Bugfix][ROCm][performance] Resolve the performance regression issue of the Qwen3-Next-80B-A3B-Thinking under rocm_atten

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32336](https://github.com/vllm-project/vllm/pull/32336) |
| 作者 | vllmellm |
| 合并日期 | 2026-01-14 |
| labels | bug, qwen, ready, rocm, v1 |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
#31712 把 ROCM_ATTN 支持的 block size 收窄到 16/32,致 block 544 模型 TTFT/吞吐暴跌,补回 544 恢复 Triton 路径。

## 问题现象
使用非标准 block size(block size 544,如 Qwen3-Next-80B-A3B-Thinking)的模型在 ROCM_ATTN 后端下出现严重性能回退:TTFT 与整体吞吐大幅下降(massive degradation)。

## 根因
正文披露:PR #31712 为防止不支持的配置,把 ROCM_ATTN 支持的 kernel block size 限制为仅 16 和 32(源自 AMD GPU 共享内存/LDS 约束),误伤了依赖更大、非标准 block size 的模型(此前由 PR #31380 优化),造成性能悬崖。

## 修复方案
`vllm/v1/attention/backends/rocm_attn.py` 的 `get_supported_kernel_block_sizes` 补回 544:

```diff
-        return [16, 32]
+        return [16, 32, 544]
```

注释说明:16/32 限制对原生 C++ kernel 合理,但非标准尺寸应经 #31380 的 Triton 路径支持——标准模型走原 bit-addressing 逻辑,非标准模型走优化后的 kernel 逻辑。

## 影响与回归风险
- 影响窗口:正文说明自 #31712 起受影响。
- 回归面:ROCM_ATTN 的 block size 支持判定;标准 16/32 模型行为不变(基准测试显示 negligible impact),block 544 模型重新走上优化路径。

## 关联
[#31712](https://github.com/vllm-project/vllm/pull/31712)、[#31380](https://github.com/vllm-project/vllm/pull/31380)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
