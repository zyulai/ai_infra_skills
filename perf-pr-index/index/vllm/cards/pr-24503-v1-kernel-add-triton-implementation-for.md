# PR #24503: [V1][Kernel] Add triton implementation for `reshape_and_cache_flash`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bringlein |
| 合并 | 2025-09-23 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/24503 |

## 一句话结论

为 KV cache 写入内核 `reshape_and_cache_flash` 增加纯 Triton 实现,使 `triton_attn` 后端摆脱对非 Triton/CUDA 自定义算子的依赖,H100/MI300 性能与 CUDA 版持平或略好。

## 动机

V1 的 `triton_attn` 注意力后端虽然注意力主体是 Triton,但 KV cache 写入仍依赖 CUDA 编译的自定义算子 `ops.reshape_and_cache_flash`,使该后端无法完全摆脱非 PyTorch/Triton 内核(对构建、分发与可移植性都是负担)。补齐这一块后 `triton_attn` 可成为纯 Triton 后端。

## 优化点

新增 `vllm/attention/ops/triton_reshape_and_cache_flash.py`:Triton 内核按 `(num_tokens, cdiv(num_heads*head_size, TILE_SIZE))` 网格启动,每个 program 处理一个 token 的一块 head 维元素,按 `slot_mapping` 定位 `(block_idx, block_offset)` 后把 key/value 写入 paged cache;`slot_idx < 0` 的 padding token 直接跳过。FP8 KV cache 支持在内核内做 `key / k_scale` 缩放(若输入已是 fp8 则跳过),并利用 `tl.store` 按目标指针元素类型的隐式转换;uint8 cache 显式 fp8 cast 不支持(triton 3.4 不支持显式 cast 到 fp8e4m3fnuz),改为先 `view(fp8_dtype)`。不使用 autotune,而是按平台启发式选 TILE_SIZE/num_warps/num_stages(HIP 4/8,CUDA 10/16,compute capability < 9 时 TILE_SIZE 上限降到 512)。仅支持 NHD layout,不支持 HND。`triton_attn.py` 的 `forward` 换用该实现,并把 fp8 `view` 提前到写入调用之前。

## 关键改动

```text
vllm/attention/ops/triton_reshape_and_cache_flash.py: 新增 Triton 内核与启动封装(fp8、NHD、启发式调参)
vllm/v1/attention/backends/triton_attn.py: forward 改调 Triton 实现,fp8 view 前移
tests/kernels/attention/test_cache.py: test_reshape_and_cache_flash 按 implementation 参数化覆盖 cuda/triton
benchmarks/kernels/benchmark_reshape_and_cache_flash.py: 增加 --implementation cuda|triton 与 --mode cudagraph|no_graph
```

关键摘录(内核主循环骨架):

```python
token_idx = tl.program_id(axis=0)
slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
if slot_idx < 0:
    # Padding token that should be ignored.
    return
block_idx = slot_idx // block_size
block_offset = slot_idx % block_size
tgt_idx = block_idx * block_stride + block_offset * page_stride
```

## 性能声明

PR 正文:"The kernel itself has the same (or slightly better) performance on H100 and MI300 as the CUDA kernel",数据以两张截图形式给出,正文未附具体数字表;benchmark 脚本默认用 CUDA graph 模式测量(也可 `--mode no_graph`)。除截图外 PR 未注明具体 µs 数值与测试条件明细。

## 验证与回归风险

单元测试 `tests/kernels/attention/test_cache.py::test_reshape_and_cache_flash` 对 cuda/triton 两种实现、auto/fp8 dtype、NHD/HND layout 全组合参数化(triton+HND 会 skip),结果 216 passed / 72 skipped;CUDA 版保留 `opcheck`。端到端:`VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` 下用 lm_eval 跑 gsm8k(5-shot, Llama-3.1-8B-Instruct, limit 500, batch auto),H100 上 main 0.798/0.782(flexible/strict) vs 本 PR 0.800/0.784,MI300 结果相同。回归风险集中在:仅 NHD layout;fp8 隐式 cast 依赖 triton 版本行为(compute capability < 9 有独立 TILE_SIZE 分支);`triton_attn` 后端的 KV 写入路径整体切换,其他注意力后端不受影响。

## 关联

同区域为 V1 `triton_attn` 后端与 KV cache 写入内核(`reshape_and_cache_flash` CUDA 版、`reshape_and_cache` 系列)。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
