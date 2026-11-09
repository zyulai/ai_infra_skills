# PR #30169: [GDN/KDA] Fuse SM100 CuteDSL prefill state I/O into the chunk h kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-07-16 |
| 分类 | Attention / Kernel (tier: major) |
| Label | blackwell, run-ci, run-ci-extra, linear-attention, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 8 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/30169 |

## 一句话结论

SM100 CuteDSL prefill 的 SSM 状态 gather/scatter 融合进 chunk h 内核 TMA 读写,去掉每层 2 次 launch 与 2 个 [N,HV,V,K] 中间张量,GDN wrapper 高并发下省 42.7%。

## 动机

SM100 CuteDSL chunk prefill wrappers(GDN `gdn_cutedsl.py`、KDA `kda_cutedsl.py`)在每次层调用时于内核外做 eager 状态 I/O:用 `ssm_states[indices].contiguous()` gather 初始状态、`index_copy_` scatter 最终状态——每方向多一次 kernel launch 加一个 `[N, HV, V, K]` 中间分配,每 linear-attn 层每 prefill step 一次(HV=32、K=V=128 时约每 seq 每层 1 MB,N=64 即每层约 64MB gather + 64MB scatter)。这类 per-call PyTorch 开销此前曾把(本就快的)CuteDSL 内核拖到 Triton 之下。栈的其他部分已做对:Triton chunk 路径把状态 I/O 融进 h kernel(indexed `h0` load、in-place indexed `ht` store),CuteDSL decode 内核也直接按 per-slot 索引读 pool——prefill 是剩下的缺口。

## 优化点

1. **h 内核池寻址**:`kernel_h`(GDN/KDA 两份,改动相同)接收 `state_indices [N]` int32,每个 sequence 以 `state_slot = state_indices[seq_id]` 解析状态行(与既有 `cu_seqlens[seq_id]` 同款运行时张量索引);H0 的 TMA load 与最终状态的 HT TMA store 直接寻址 `tmaH0/tmaHT[state_slot, head_id]`。launch grid 的序列数从 `h0.shape[0]` 改为 `cu_seqlens.shape[0] - 1`;编译期 fake tensor 给 `h0/ht` 独立于序列数的池大小符号维;dtype 处理不变(bf16/fp32 pool 均支持,cast 融合在既有 store 路径)。
2. **双模式入口**:`chunk_kda_cutedsl`/`chunk_gated_delta_rule_cutedsl` 新增可选 `h0_indices`/`initial_state_indices`。池模式:h kernel 原地读写 pool 行,返回的 final state 即 pool 本身;稠密模式(不给 indices):内部 `arange` 逐位保持旧的"返回全新 ht"契约,既有调用方与测试零改动。
3. **wrapper 简化**:`gdn_cutedsl.py`/`kda_cutedsl.py` 的 extend 直接传 pool + int32 slots,删掉 gather、`.contiguous()` 与 `index_copy_` scatter;padding slot(-1)仍重映射到 sentinel(最后一个)pool slot,与 decode 内核一致。净效果:每层每 prefill 调用少 2 次 launch、少 2 个 `[N, HV, V, K]` 中间张量,状态往返经 fresh allocation 的流量归零。

## 关键改动

```text
python/sglang/kernels/ops/attention/linear/{gdn,kda}_blackwell/kernel_h.py:
  h 内核增加 state_indices 参数,TMA load/store 按池槽寻址,grid 改由
  cu_seqlens 决定;compile 的 fake tensor 增加 num_state_slots 维
python/sglang/kernels/ops/attention/linear/{gdn,kda}_blackwell/__init__.py:
  chunk 入口增加 initial_state_indices/h0_indices,双模式分派
python/sglang/srt/layers/attention/linear/kernels/{gdn,kda}_cutedsl.py:
  extend 删除 gather/.contiguous()/index_copy_,传池与 slots
test/registered/attention/test_{gdn,kda}_prefill_cutedsl.py: 新增
  pool_mode_matches_dense 逐位相等测试(bf16 + fp32)
```

关键片段(diff 摘录,KDA 入口双模式):

```python
if h0_indices is None:
    ht = torch.empty_like(h0)  # dense: 保留返回全新 ht 的契约
    state_indices = torch.arange(...)
else:
    ht = h0                    # pool: 原地读写池行
    state_indices = h0_indices
```

## 性能声明

PR 正文,未独立核实。B200 wrapper 级基准,旧(gather + dense chunk + scatter)vs 新(pool-indexed chunk),Hv=32、K=V=128、bf16 pool,固定 8K token 预算按 N 均分;`g+s` 为独立 gather+scatter 成本(节省的理论上限):

| N | len | KDA old→new ms | KDA saved | GDN old→new ms | GDN saved | g+s ms |
|---|---|---|---|---|---|---|
| 8 | 1024 | 1.090→1.065 | 2.3% | 0.241→0.218 | 9.8% | 0.019 |
| 16 | 512 | 1.109→1.071 | 3.4% | 0.259→0.224 | 13.5% | 0.029 |
| 64 | 128 | 1.215→1.099 | 9.5% | 0.364→0.249 | 31.5% | 0.105 |
| 128 | 64 | 1.353→1.134 | 16.2% | 0.510→0.292 | **42.7%** | 0.215 |

作者分析:节省量与独立 gather+scatter 成本几乎逐点吻合(N=128 时省 0.219ms vs g+s 0.215ms),即状态 I/O 被完整消除;相对收益随并发增长(状态 I/O 按 sequence、计算按 token 计),且 GDN 收益更大——其 chunk 流水线在同等 token 预算下比 KDA 快约 4.5×,I/O 占比因此更高。

## 验证与回归风险

- 新增 `test_{kda,gdn}_chunk_cutedsl_pool_mode_matches_dense`(bf16 + fp32 两种 pool dtype):池模式必须逐位复现稠密路径——`o` 相等、索引槽的最终状态行相等、所有未索引池行原样(`assert ht_pool is pool` + `torch.equal`),因仅寻址不同、内核与数学相同,用精确相等而非容差。
- 既有对 token-by-token 递推参考的正确性测试经稠密模式路径不变地覆盖新签名。
- 两个测试文件均注册在 `4-gpu-b200` runner(`base-b-kernel-unit`)。CI 状态(pack 记录):Base 与 Extra 均通过。
- 回归面:仅 SM100 CuteDSL prefill 路径;稠密模式的 `arange` 保留旧契约,外部调用方不受影响;风险集中在池模式的 slot 重映射(-1 → sentinel 槽)与 in-place 写回与投机解码 rollback 的交互(wrapper 层 `target_verify` 签名未在 diff 中展开)。

## 关联

#29472(同作者、同 batch,FlashKDA safe-gate KDA prefill 后端,同为 linear-attention prefill 内核方向);本卡 label 与 #28185(GDN/KDA 状态池)同区域,pack 未直接引用。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
