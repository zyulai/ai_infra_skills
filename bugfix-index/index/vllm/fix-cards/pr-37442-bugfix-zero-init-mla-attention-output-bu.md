# PR #37442: [Bugfix] Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37442](https://github.com/vllm-project/vllm/pull/37442) |
| 作者 | elvircrn |
| 合并日期 | 2026-03-19 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
CUDA graph padding 槽位 seq_lens=0 时 MLA kernel 不写输出,残留 NaN 经 per-tensor FP8 量化 amax 传染真实 token;按层预分配零初始化输出缓冲修复。

## 问题现象
CUDA graph decode 带 padding 时(如 batch 1024 中只有 1 个真实请求),未使用槽位的 `seq_lens=0`,MLA decode kernel 对这些槽位不写输出,输出缓冲中残留旧数据;若旧数据含 NaN(来自上一次迭代或未初始化内存),会经下游 per-tensor FP8 量化的 `amax`(对整个 batch 归约)传染到真实 token。生产环境观察到:GB200(SM100)上跑 DeepSeek-R1 NVFP4 出现间歇性 NaN 输出。最小复现确认:脏输出缓冲 + seq_lens=0 → NaN 传播;零初始化输出 → 干净。

## 根因
PR 正文披露两种在 padding 槽位产生 NaN 的机制:
1. **kernel 跳过(seq_lens=0)**:kernel 对 padding 行不写任何输出,输出缓冲中已有内容原样保留。
2. **TMA tile 越界读(seq_lens=1)**:MLA kernel 按 128 条目的 TMA tile 读 KV cache,seq_lens=1 时第 1–127 条目从未初始化的 KV cache 页读入,若含 NaN 则 softmax 产生 NaN(关联 flash-attention issue #1974)。

vLLM 已对 KV cache 页做零初始化(已修机制 2);机制 1 则要求输出缓冲本身干净——原实现用 `new_empty` 分配输出,内容未定义。

## 修复方案
每个 backend 在层初始化时预分配零初始化输出缓冲(缓存为实例属性 `_decode_out`),每次调用按需切片复用,避免每次 CUDA graph replay 都跑 memset:

```diff
-        out = q_nope.new_empty((B_q, MAX_HEADS, D_latent), dtype=dtype)
+        if (
+            self._decode_out is None
+            or self._decode_out.shape[0] < B_q
+            or self._decode_out.dtype != dtype
+        ):
+            self._decode_out = q_nope.new_zeros((B_q, MAX_HEADS, D_latent), dtype=dtype)
+        out = self._decode_out[:B_q]
```

- CUTLASS MLA(`cutlass_mla.py`):上述模式,首次调用惰性分配。
- FlashInfer MLA(`flashinfer_mla.py`):因 FlashInfer 的 `out=` 校验硬编码 3D 形状而 kernel 在 q_len>1 时写 4D(上游 bug),仅在单 token decode(q_len==1)时传预分配的 `out=`;q_len>1 时在 kernel 返回后把 padding 槽位清零(`o[num_real:] = 0`,`num_real` 来自 `attn_metadata.num_decodes`)。正文留有 TODO:上游修复 FlashInfer。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是使用 CUTLASS MLA 或 FlashInfer TRT-LLM MLA decode + CUDA graph padding + 下游 per-tensor FP8/NVFP4 量化的配置。
- 回归面:两个 MLA backend 的 decode 输出缓冲生命周期变化——缓冲按层持久化复用,容量不足或 dtype 变化时才重新分配;若有形状/设备变化未覆盖的分支,可能复用过期缓冲(q_len>1 的 FlashInfer 路径改为 kernel 后清零,依赖 `attn_metadata.num_decodes` 的正确性)。性能:基准显示零开销(168 µs/iter 修复前后相同);GB200 + DeepSeek-R1 NVFP4(1 真实 + 1023 padding,~1200 seq_len)验证 NaN 消失。

## 关联
- https://github.com/Dao-AILab/flash-attention/issues/1974(机制 2 相关)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
