# PR #46404: [DeepEP V2] Bound num_max_tokens_per_rank in do_expand=False

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-06-23 |
| 分类 | MoE(major) |
| Label | bug, ready |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 1 个源码文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/46404 |

## 一句话结论

修复 DeepEP v2 decode 路径 recv buffer 按 worst case R×max_num_batched_tokens 取尺寸的问题,GB200×4 上 c1 decode step time 100.7ms → 16.4ms(6.1×)。

## 动机

do_expand=False(decode/cudagraph)模式下,DeepEPV2PrepareAndFinalize._do_dispatch 未设置 num_max_tokens_per_rank,ElasticBuffer.dispatch 回退到 buffer 初始化容量(= max_num_batched_tokens)。recv buffer 因此按 worst case R×max_num_batched_tokens 分配,expert 内核即使只有少量 decode token 也要处理约 R×8192 行,成为 decode step 时间的主导项。

## 优化点

decode 路径把 num_max_tokens_per_rank 绑定到 DP-padded batch size(max(num_tokens_across_dp),各 rank 一致)并向上取 2 的幂。pow2 bucketing 是关键:DeepEP JIT 对每个不同的 num_max_tokens_per_rank 编译一个 dispatch kernel,直接喂 per-step 原始尺寸会每个 batch size 重编译一次(cicc 重编译风暴,高并发下饿死 GPU);pow2 把编译集合限制到约 log2(max_num_batched_tokens) 个值(编译一次后缓存),decode 侧保持小(如 1 token → 1),prefill 侧封顶在 buffer 初始容量。DP 元数据缺失时回退 tokens.shape[0]。prefill(do_expand=True)保持原 None/CPU-sync 路径不变。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py:
  _do_dispatch 为 do_expand=False 计算 pow2 化的 num_max_tokens_per_rank
  并传入 ElasticBuffer.dispatch
```

```python
num_max_tokens_per_rank = None
if not do_expand:
    dp_meta = get_forward_context().dp_metadata
    if dp_meta is not None:
        n = int(dp_meta.num_tokens_across_dp_cpu.max())
    else:
        n = tokens.shape[0]
    num_max_tokens_per_rank = 1 << max(n - 1, 0).bit_length()
```

## 性能声明

DeepSeek-V4-Flash,GB200 ×4,dp=4 -ep,DeepEP v2 + Triton MoE:concurrency=1 decode step time 100.7 ms → 16.4 ms(6.1×);concurrency=64 不再出现 cicc 重编译风暴(此前风暴期间 GPU 利用率约 0%);concurrency=1024 吞吐约 +6%;GSM8K 精度不变(约 95%)。

## 验证与回归风险

ruff check、ruff format、mypy(pre-commit)通过;未新增功能测试。风险限于 DeepEP v2 decode dispatch 的 recv buffer 尺寸选择:pow2 化会让 decode buffer 比实际 batch 略大(最多 2 倍),prefill 路径与 DP 元数据缺省回退不变。

## 关联

正文排查过相关 open PR:#45282(NVFP4 dispatch)、#40718(combine_v2)、#45193(topk_ids optional for do_expand)、#45321(Dockerfile 版本),均不触及 decode recv-buffer 尺寸。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
