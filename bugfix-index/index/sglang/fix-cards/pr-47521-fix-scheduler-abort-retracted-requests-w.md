# PR #47521: [Fix] scheduler: Abort retracted requests whose KV the recompute cannot rebuild

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47521](https://github.com/sgl-project/sglang/pull/47521) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-10-30 |
| labels | run-ci, apple-silicon |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
retract 后 recompute 重建不了 KV 的请求(multimodal/input_embeds/embed overrides)此前默默产出错误或乱码输出;`release_req` 统一判定可否恢复,不可恢复即以 503 abort,另统一 PD fill-length 公式并让 /detokenize 对越界 id 返回 400。

## 问题现象
两类"recompute 无法重建 KV"的 retract 场景不报错而产出坏结果:
- PD decode rebootstrap(`pause_generation(mode="retract")` 及 DeepSeek-V4 经 `backup_kv_cache` 的 retract):`Req.build_rebootstrap_payload` 只携带 token ids,multimodal、`input_embeds`、`positional_embed_overrides` 请求被用错误 KV 重算,产出错误输出且无任何报错;
- 本地 re-prefill(非分离 retract):`input_embeds` 只覆盖 prompt,`reset_for_retract` 会重启生成;在输出已下行(流式已发 token,或非流式请求越过 `SGLANG_FORCE_STREAM_INTERVAL` flush)后,重启的 token 会被拼接在陈旧的 `send_token_offset` 之后,生成内容错乱。
附带:`/detokenize` 对负数或超 u32 的 id 返回 500,对 >= vocab_size 的 id 静默丢弃并返回 200。

## 根因
正文披露:retract 的恢复机制各有前提——rebootstrap 只能从纯 token ids 重建 KV,故 multimodal/embeds/overrides 请求不可恢复;re-prefill 会重启 `input_embeds` 请求的生成,而一旦输出已发出,重启对下游不可见且拼接错位。原代码没有统一的"能否恢复"判定:各 retract 路径只检查 KV backup 是否成功,不检查请求负载是否可被重建。

## 修复方案
- `schedule_batch.py`:`Req` 新增 `resumable_by_rebootstrap`(无 multimodal_inputs/input_embeds/positional_embed_overrides)与 `resumable_by_reprefill`(无 input_embeds 或 `send_token_offset == 0`)两个属性;`release_req` 成为所有 retract 的唯一判定点,按 `disaggregation_mode`/`offload_kv` 分支返回 `resumable`;`retract_all` 返回不可恢复请求列表;新增 `unresumable_abort_message()`。
- `scheduler.py::pause_generation`:不可恢复请求以 `FINISH_ABORT(..., 503)` 经 `send_abort_output` 终止;`retract_decode` 对不可恢复请求按非 decode 模式给出 503 消息;`mem_cache/common.py::backup_kv_cache` 复用 `resumable_by_rebootstrap`(原仅挡 multimodal)。
- 每次 PD rebootstrap 现在至少有一个 output id,transfer commit 的 `resumes_output` 改为 `decode_req.is_rebootstrap` 加 assert。
- PD fill-length 公式 `len(origin_input_ids) + max(len(output_ids) - 1, 0)` 收敛到 `disaggregation/utils.py::pre_alloc_fill_len`,替换 `DecodePreallocQueue._pre_alloc_fill_len` 及预构建 seq_len 两处重复(无行为变化)。
- `serving_tokenize.py`:`/detokenize` 新增 `_out_of_vocab_error`,越界 id 返回 400。

关键 diff(schedule_batch.py):

```python
    @property
    def resumable_by_rebootstrap(self) -> bool:
        """A rebootstrap recomputes the KV from token ids alone."""
        return (
            self.multimodal_inputs is None
            and self.input_embeds is None
            and self.positional_embed_overrides is None
        )
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面(正文 "Behavior changes" 原文归纳):上述 abort 取代 PD decode 下"默默错误输出"与本地 `input_embeds` retract 输出已发后的"乱码重启";`/detokenize` 原返回 500 或静默丢弃的场景改返回 400;priority preemption 对被抢占 `input_embeds` 请求仍从 prompt 重 prefill,该路径不变;retract 后仍可恢复的请求(未发送输出的 input_embeds、纯文本请求)继续按原路径 rebootstrap/re-prefill。单测覆盖 pause_generation 下 decode/非 decode 两分支的 abort/hold/requeue 组合。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
