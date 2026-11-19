# PR #67489: [Bugfix] Fix OOM from unused FA3 MLA split-KV buffer on decode batches above CUDA graph sizes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#67489](https://github.com/vllm-project/vllm/pull/67489) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-11-17 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 8 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | resource |

## 一句话结论
超过最大已捕获 CUDA graph 尺寸的 decode 路径批次在 `_build_decode` 里传了 `num_splits=0`,FA3 把 0 理解为"自行选择",按"一个请求持有全部 query"规划 split 数并分配每层约 1.25 GiB 的 fp32 split-KV 累加器,而 16 头时运行时实际只用 1 个 split——纯浪费的分配在可扩展 KV cache 的精确预留模型下触发 OOM,nightly DeepSeek-V2-Lite prefetch offload 精度记 0.000;修复令这些批次传 `num_splits=1`。

## 问题现象
1×H100 上开启 prefetch offload 跑 DeepSeek-V2-Lite,首个大 gsm8k step 在 FA3 MLA attention 调用内 GPU 内存耗尽:`RuntimeError: CUDA out of memory. Tried to allocate 1.25 GiB ... 27.93 GiB allowed`,引擎死亡,nightly 检查精度 0.000。基础 commit 上 5 次运行 3 次失败(是否失败取决于请求时序决定的批次形状);可稳定复现的检查中,170 prompts × 48 新 tokens 叠加 640-token 缓存前缀在 `cudagraph_mode=PIECEWISE` 下申请 2.46 GiB 也 OOM。修复后 nightly 3/3 通过(精度 0.335/0.330/0.345),首个 eval step(7796 decode-path tokens、74 requests)拿到 `num_splits=1`,reserved 峰值 27.39 GiB 低于 27.93 GiB 上限。

## 根因
正文披露根因链:

- `FlashAttnMLAMetadataBuilder.reorder_batch_threshold = 512`,新 token 数不超过 512 的请求都走 MQA decode 路径;gsm8k 5-shot 的大量短 prefill 叠加缓存前缀会形成约 8000 token 的 decode 路径批次。
- 超过最大捕获 graph 尺寸(此处 512)时,`_build_decode` 传 `num_splits=0`(所有 graph 模式如此)。对 FA3,`0` 不是"不分",而是"FA3 自己选":varlen 批次按"单个请求持有全部 query"规划 split 数,每层分配 `splits × tokens × heads × 512 × 4` 字节的 fp32 累加器;失败 step 为 5 splits 约 1.25 GiB,而 16 头时 FA3 运行时只用 1 个 split。
- 没有 warmup 跑过该形状,可扩展 KV cache 只为实测瞬态(1.42 GiB)加余量留白;一旦更长寿的 tensor 落进释放的累加器块,分配器需要第二个同尺寸块,该 step reserved 内存涨 2.9 GiB,触及上限。可扩展 KV cache 之前,`gpu_memory_utilization=0.92` 剩下的约 6 GiB 空闲掩盖了这一浪费。

## 修复方案
改 `vllm/v1/attention/backends/mla/flashattn_mla.py` 的 `_build_decode`,门控变为 `if self.max_cudagraph_size:`:

```python
max_num_splits = 1
if not envs.VLLM_BATCH_INVARIANT:
    if not self.max_cudagraph_size:
        max_num_splits = 0
    elif num_decode_tokens <= self.max_cudagraph_size:
        max_num_splits = self.max_num_splits
```

捕获尺寸内的批次行为不变(full graph 用 `self.max_num_splits`,PIECEWISE 用 0);更大的批次取 `num_splits=1`;`enforce_eager` 或 `cudagraph_mode=NONE` 完全不变。KV cache 尺寸不变(26766 blocks 前后一致)。新增单测 `test_flash_attn_mla_eager_decode_batch_allocates_no_split_accumulator`(FULL_AND_PIECEWISE 与 PIECEWISE 各一例),断言瞬态小于单 split 累加器大小,基础 commit 上 2 failed(`assert 2680138752 < ...`),修复后 2 passed。

## 影响与回归风险
- 内存收益:失败 step 的无用 split-KV 缓冲为 1.2 GiB,同一 8192-token 预算内其他形状最高 6.5 GiB;16 头时瞬态从 2.496 GiB 降到 0.001 GiB(170×48 over 640-token 前缀)。
- 性能权衡(FA3 kernel time,num_splits=1 vs 旧 0):16 头/rank 最多 1.01x,输出位级一致;8 头/rank 0.71x-1.00x,输出差一个 bf16 ulp 以内;4 头/rank 下少数"少请求长 query 长上下文"形状变慢(8×70 tokens 在 1900/8192/32768 KV 为 1.51x/2.30x/2.68x),作者保留简单规则以避免复制 FA3 启发式。
- 明确遗留:① `enforce_eager`/NONE 无捕获尺寸可比,同一 170×48 负载仍 OOM,需另立规则(如 splits×tokens 上限);② prefill graph 捕获到 8192 tokens 时走图内分支 32 splits,前后都 OOM(申请 7.83 GiB);③ `vllm/attention/backends/flash_attn.py` 存在同样的 `max_num_splits = 0` 块,测得 Qwen3 GQA 形状的无用缓冲但无端到端 OOM,未改;④ DCP 未测试,仅单 GPU。
- 对照 #67296:其 sizing 步骤从不跑大型 MLA decode 批次,预留增长(0.77 GiB)低于已保留的 1.42 GiB,失败 step 依旧在同一 FA3 MLA 调用 OOM;本 PR 新测试在其改动上同样失败。

## 关联
- Fixes #67307;与 #67296 的关系见影响与回归风险。
- 涉及文件:`vllm/v1/attention/backends/mla/flashattn_mla.py`、`tests/v1/attention/test_mla_backends.py`(该测试文件已在 `.buildkite/test_areas/attention.yaml` 中)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
