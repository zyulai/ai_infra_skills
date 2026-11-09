# PR #58454: [Bugfix][GLM-5.3-Flash] kpool corruption with speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58454](https://github.com/vllm-project/vllm/pull/58454) |
| 作者 | mmastrac |
| 合并日期 | 2026-09-25 |
| labels | bug, ready, glm |
| 热度 | 👍 0 · 💬 34 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
被拒绝的 pool 补全 draft 会覆写 kpool 尾环中已提交的 key,导致长文本摘要 key 损坏;修复为把尾环扩容到 kpool*cdiv(kpool+num_spec,kpool) 槽位。

## 问题现象
长 GLM-5.3-Flash 推理 trace 中出现随机性损坏。影响所有 `num_speculative_tokens >= 2` 的投机解码方法;只有 decode 阶段(模型自身输出)构建的 pool 受影响,且仅在上下文超过 `index_topk`(2048)、多个 pool 竞争选择时才显现。最可能的表现是长推理链中后段思考回指前段时精度略降,可能造成「凭记忆转写」类失败。

## 根因
正文披露:被拒绝的 pool-completing draft 会暗中破坏 pool 选择逻辑。当一个补全 pool 的 draft 被拒绝时,排在它后面被 stash 的 draft 已经覆写了该 pool 已提交的 key。一个由 4 个 token 组成的 pool 因此拿到错误的 summary key,indexer 可能跳过这些 token 或选中用处较小的 pool。作者的独立复现显示:修复前(RING=4)`128 个 fp8 key 字节中 106 个`与无投机参考不一致;修复后(RING=8)完全一致。

## 修复方案
把每请求的 tail ring 从 `kpool` 个槽扩容为 `kpool * cdiv(kpool + num_spec, kpool)` 个槽,seed 与 decode kernel 统一改用 `pos % RING` / `tslot // RING` 寻址:
- `Glm5NextTailCache.get_kv_cache_spec`(vllm/models/glm5next/common/attention.py)计算 `span = _index_kpool + num_speculative_tokens`,`ring = _index_kpool * next_power_of_2(cdiv(span, _index_kpool))`,`KpoolTailSpec` 的 `block_size`/`sliding_window` 改为 ring,并断言 `cache_config.block_size % ring == 0`。
- nvidia 与 amd 两份 `kpool_compress.py` 的 `_kpool_tail_seed_kernel`、`_kpool_decode_update_batched_kernel` 新增 `RING` constexpr,替换原先按 `POOL_SIZE` 的取模与除法;host 侧断言改为 `ring >= pool_size and ring % pool_size == 0`。
- 新增内核级与 tail-slot-mapping 回归测试。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:GLM-5.3-Flash(GLM5Next)kpool indexer 的 tail cache 尺寸与寻址全部改变——tail cache 显存占用随 num_spec 增大(最大约 4 倍),`KpoolTailSpec.block_size` 变化影响 attention block 对齐(测试注释指出旧实现 7 个 draft 时 ring=12 不整除 640 块导致前缀缓存命中降到 1920 的倍数,由 diff 推断该问题一并修复);非 GLM5Next 模型不受影响。正文称这是 #56868/#56605 的部分修复,后续还有 PR。

## 关联
#56868, #56605, #58704, #55219

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
