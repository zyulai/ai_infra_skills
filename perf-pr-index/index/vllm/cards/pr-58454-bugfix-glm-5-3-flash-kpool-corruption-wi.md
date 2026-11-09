# PR #58454: [Bugfix][GLM-5.3-Flash] kpool corruption with speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmastrac |
| 合并 | 2026-09-25 |
| 分类 | Speculative Decoding (tier: major) |
| Label | bug, ready, glm |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/58454 |

## 一句话结论

修复 GLM-5.3-Flash kpool indexer 在 speculative decoding 下被拒绝 draft 覆写 tail ring 造成的 key 损坏。

## 动机

PR 之前,speculative decoding 中被拒绝的 pool-completing draft 会隐式破坏 pool 选择逻辑:draft 在 acceptance 未知前就 stash 进 per-request tail ring,被拒绝后,排在其后的 drafts 已经覆写该 pool 已提交的 committed keys,导致 pool 得到错误的 summary key,indexer 可能跳过这些 token 或选中较低效的 pool。影响所有 `num_speculative_tokens >= 2` 的 speculative 方法,且只影响 decode 期构建的 pool、只在 context 超过 `index_topk`(2048)、pool 之间开始竞争选择后出现。作者指出长 reasoning trace 中后续思考回溯先前思考的远距离自引用质量因此变差,可能是 re-transcription from memory 失败的诱因。

## 优化点

核心机制是把 tail ring 从恰好 `kpool` 个 slot 扩大为 `ring = kpool * cdiv(kpool + num_spec, kpool)`,seed/decode kernel 的寻址由 `pos % KPOOL` 改为 `pos % RING`,tail block 划分由 `// KPOOL` 改为 `// RING`。`Glm5NextTailCache.get_kv_cache_spec` 按 num_speculative_tokens 计算 ring 并写入 `KpoolTailSpec` 的 block_size/sliding_window,并断言 `cache_config.block_size % ring == 0`(测试注明 ring 不整除 640 的 attention block 时 scheduler block 会增长到 lcm,prefix-cache 命中降到 1920 的倍数)。NVIDIA 与 AMD 两份 Triton kernel 同步修改,host 侧断言放宽为 `ring >= pool_size 且 ring % pool_size == 0`。

## 关键改动

```text
vllm/models/glm5next/common/attention.py: get_kv_cache_spec 计算 ring 大小,写入 KpoolTailSpec 并断言整除
vllm/models/glm5next/nvidia/ops/kpool_compress.py: seed/decode kernel 新增 RING constexpr,寻址 % RING
vllm/models/glm5next/amd/ops/kpool_compress.py: AMD 侧 kernel 同步改用 RING
tests/v1/attention/test_kpool_tail_slot_mapping.py: ring 尺寸与 rejected-draft corruption 回归测试
tests/kernels/test_kpool_decode_update_batched.py: kernel 级 redo 一致性回归测试
```

关键计算:

```python
span = self._index_kpool + vllm_config.num_speculative_tokens
ring = self._index_kpool * next_power_of_2(cdiv(span, self._index_kpool))
assert self.cache_config.block_size % ring == 0
```

## 性能声明

PR 未给出端到端性能数字。正确性 repro(kernel 级):`main` ring=4 时 pool 与无推测 reference 相比 128 个 fp8 key bytes 中 106 个不同;ring=8 直接 AssertionError;本 PR ring=8(`4 * cdiv(4+3, 4)`)时与 reference 完全一致。收益定性:避免 indexer 跳过 token 或选择低效 pool,改善长 trace 的远距离自引用。PR 未注明测试条件。

## 验证与回归风险

新增 `test_rejected_draft_redo_needs_ring_slots`、`test_rejected_completing_draft_needs_ring_slots`、`test_tail_ring_divides_the_attention_block`(参数化 num_spec=0/1/4/7/13 对应 ring 4/8/8/16/32)。风险面:GLM-5.3-Flash(glm5next)tail cache 布局与 KV cache spec 变化,`KpoolTailSpec.block_size/sliding_window` 改为 ring,要求 cache block_size 为 ring 倍数,可能影响 prefix caching 与 PD connector 传递 tail cache 的路径;NVIDIA/AMD 两份 kernel 需保持一致。作者注明这是 #56868/#56605 的部分修复,后续还有 #58704 等 followup。

## 关联

同区域:#58704(同一问题的另一处修复)、#55219(本修复基于其工作)、问题单 #56868、#56605。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
