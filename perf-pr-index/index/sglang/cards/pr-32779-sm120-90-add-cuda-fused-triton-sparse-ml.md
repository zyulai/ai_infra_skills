# PR #32779: [SM120&90] Add CUDA fused Triton sparse-MLA prefill backend for DSA

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yunyang1999 |
| 合并 | 2026-10-06 |
| 分类 | MoE(major;实质为 DSA attention prefill 后端) |
| Label | documentation, performance, run-ci, jit-kernel, GLM, bypass-fail-fast |
| Diff | 素材未提供 diffstat(diff 含 7 个文件,另有 5 个文件因体积省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/32779 |

## 一句话结论

为 DSA prefill 新增融合 Triton sparse-MLA 后端:SM90 内核级 geomean 4.41x(union2),78 层 GLM-5.2 端到端最高 2.08x。

## 动机

CUDA 上 DSA prefill 没有融合 Triton 路径。`flash_mla_sparse_fwd` 要求 `num_heads % 64`(SM90)/ `% 128`(SM100+),`_forward_flashmla_sparse` 因此对 head 维做 zero-pad:GLM-5.x 在 TP8 下只有 8 个真实 head,SM90 上 7/8 的 tensor-core 工作耗在 padding 上(SM100+ 为 15/16);且 split 形式还要把 `O(tokens x splits)` 个 partial 写回 HBM 供第二个 kernel merge。main 对 SM90 FP8 KV cache 自动选择的 `flashmla_kv` 在 4096 tokens 以上更慢。这两种开销在小 head 数下都可避免。

## 优化点

新后端 `--dsa-prefill-backend triton_sparse_mla`(默认关闭):每个 query token 一个 Triton program,online softmax 保持在寄存器中,head 维只 pad 进 16 行 MMA tile(8→16)而非 pad 进 grid(8→128),`V` 直接读 `kv[:, :d_v]`,无 partial、无 merge;是代数等价改写而非近似。可选 `--dsa-triton-union {2,4}`:DSA indexer 的 topk 在相邻 token 间高度重叠(GLM-5.1 实采 trace 上 2 token 的并集仅 1.00-1.12x 单 token 行数),让 G 个相邻 token 共享一次 gather 的索引集(device 端按 token 排序后用向量化 `lower_bound` 去重,O(G*K*log K),行数 ~1.1x),`G*num_heads` 行 MMA tile 装真实工作;每行 ownership bitmask 恢复各 token 自己的 softmax 支撑,结果与逐 token 路径一致(仅累加顺序不同)。union tile 要求 `num_q_heads * union` 为 [16,32] 内的 2 的幂;在 `--enable-deterministic-inference` 与 stream capture 下强制关闭。工程健壮性:int64 token offset 防大 batch 溢出、shared-memory tile 逐步降级并按 shape 记住可用 tile、启动校验器拒绝 ROCm/SM<90/每 rank 超 32 head/`index_kpool>1`/非法 union;非 bf16 KV cache 直接报错。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_prefill.py:
  新模块,per-token kernel + union dedup kernel + union kernel(511 行)
python/sglang/srt/layers/attention/dsa_backend.py:
  forward_extend 分发、_forward_triton_sparse_mla、启动校验、RAGGED 路由
python/sglang/srt/arg_groups/fields/exec_.py: 新参数 --dsa-triton-union {0,2,4}
python/sglang/kernels/ops/attention/flash_mla_sm120.py:
  SM120 GLM-FP8 allowlist 收录 triton_sparse_mla(prefill)
test/registered/kernels/benchmark/attention/bench_dsa_triton_sparse_mla_prefill.py:
  校准随机游走索引的 benchmark(CI 注册)
```

per-token kernel 核心(节选):

```python
t64 = t.to(tl.int64)  # h=128 时 q 偏移从 t=29128 起会越过 2^31
...
qk = tl.dot(q_main, tl.trans(kv_main))
qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
qk = tl.where(valid[None, :], qk, -float("inf"))
```

## 性能声明

内核级数字用真实捕获的 GLM-5.1 索引集,5 个 (batch, seqlen) 点,15 次取 median,每点先过 fp32 参考 cosine 1.0000 门禁:

- SM90 vs `flash_mla_sparse_fwd`(head 8→64 pad):geomean per-token 3.12x、+union2 4.41x、+union4 4.06x;如 bs1x8192:17.632ms → 5.173ms(union2)
- SM120:union 相对 per-token 路径 1.69-1.90x
- SM90 端到端,GLM-5.2(78 层,真实 FP8 权重,TP8):4096 tokens 0.888s → 0.516s(union2,1.72x;对 main 自动选择的 `flashmla_kv` 2.08x);8192 tokens 1.891s → 1.048s(1.80x;2.01x vs auto);2048 tokens 不变——`max_kv_len <= 2048`(默认阈值)时路由到 dense one-shot 路径,内核不被调用
- SM120 端到端,GLM-5.2-NVFP4,TP8,基线为 main 自动选择的 `flashinfer_sparse_mla`:union4 从 4096 tokens 起端到端 1.13-1.14x(如 8192:2315.6 → 2034.8ms);per-token 路径端到端约 1.01x
- 8 层 dummy-weight 配置(SM90,TP8):GLM-5.2 126.3ms(auto)→ 70.3ms(union2),vs `flashmla_sparse` 1.53x

注:PR 中"~14% faster"指 main 上参考路径自 7 月以来变快约 14%(解释比率从 1.6x 降到 1.4x),不是本 PR 的提升数字。

## 验证与回归风险

三层验证:内核级 `test_dsa_triton_sparse_mla_prefill.py`(fp32 参考门禁、union 精确性含 T%G 尾部、去重契约、tile 降级、int32/int64 等价、确定性);CPU 单测 20 例(校验器边界、deterministic/capture/breakable-graph 门禁、RAGGED 路由回归、SM120 auto-selection 不受影响);模型级(a)layer-0 attention 输出 fp64 对比——对 bf16 参考在重跑噪声 1.5-2x 内,(b)GLM-5.2 FP8 全词表 next-token logits——per-token 路径在 rerun 控制组噪声带内,argmax 4096 一致、8192 上 union2 翻转(与 main 自己的替代 kernel 行为同阶),(c)SM120 NVFP4 首 token 一致性 47/50(算术差异:本 kernel 走反量化 bf16,flashinfer 直接消费 FP8 pool),union 与 per-token 在所有 prompt 上 bit-identical。已知缺口:`--enable-mixed-chunk` + FP8 KV cache 会经 PAGED transform 撞 bf16 报错,main 上 `flashmla_sparse` 有同样问题,留待单独修复。CI:PR Test (Base) 失败、AMD ROCm 10 失败。

## 关联

模块 docstring 提到 gfx950-only 的 FP8 姊妹模块 `dsa/triton_sparse_mla.py` 已存在;对比基线为 `flashmla_sparse` / `flashmla_kv` / `flashinfer_sparse_mla`;已知后续修复(mixed-chunk 路由)与 `flashmla_sparse` 一并处理。同分类其他 PR 编号素材未列出,写"无"。
