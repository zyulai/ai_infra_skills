# PR #28221: Fix EagleDraftExtendInput missing kv_indptr crash with triton/DP attention

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28221](https://github.com/sgl-project/sglang/pull/28221) |
| 作者 | yctseng0211 |
| 合并日期 | 2026-06-17 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论

#24860 把 draft-extend 的 spec_info 换成未定义 `kv_indptr` 的 `EagleDraftExtendInput`,多个 attention backend 读该属性即崩;补上默认为 None 的字段。

## 问题现象

EAGLE3 + DP attention 下,空闲的 DP rank 走 `forward_idle` -> `init_forward_metadata` 时崩溃,`test/registered/spec/eagle/test_eagle_dp_attention.py` 失败,scheduler 直接挂掉。报错(照抄原文):

```
    File ".../layers/attention/triton_backend.py", line 570, in init_forward_metadata
        if spec_info is None or spec_info.kv_indptr is None:
    AttributeError: 'EagleDraftExtendInput' object has no attribute 'kv_indptr'
```

该问题在 AMD CI 暴露,因为那条 lane 使用 `--attention-backend triton`,而 CUDA lane 用 `fa3`(唯一不访问 `spec_info.kv_indptr` 的 backend)。同样的崩溃会发生在任何使用 `triton`、`flashinfer`、`flashinfer_mla`、`aiter` 或 `wave` backend 的平台上。

## 根因

正文披露:PR #24860("[Spec] Install `EagleDraftExtendInput` as the V2 draft-extend `spec_info`")把 draft-extend 的 `spec_info` 从 `EagleDraftInput` 换成新的 `EagleDraftExtendInput`,但后者没有定义 `kv_indptr` 属性;而多个 attention backend 在 `init_forward_metadata` 的 decode/idle 路径直接读取 `spec_info.kv_indptr`,于是抛 `AttributeError`。

## 修复方案

给 `python/sglang/srt/speculative/eagle_info.py` 的 `EagleDraftExtendInput` 增加字段,恢复旧 `EagleDraftInput` 提供的接口:

```diff
+    # None for draft-extend's idle batch; attention backends fall back to
+    # rebuilding plain metadata from seq_lens when this is None.
+    kv_indptr: torch.Tensor = None
```

为 None(draft-extend idle batch 场景)时,各 attention backend 回退到用 `seq_lens` 重建普通 metadata,与 a52ccd2 之前的行为一致;一个字段同时修好所有受影响 backend。

## 影响与回归风险

- 影响窗口:自 #24860 引入 `EagleDraftExtendInput` 起,至本修复。
- 回归面:仅 `EagleDraftExtendInput` 的 dataclass 字段定义,新增字段默认 None,不改变既有赋值路径;受影响的是上述 5 个 backend 的 idle/decode 元数据构建,现恢复为读 `kv_indptr`(为 None 时重建),`fa3` 路径不受影响。EAGLE3 + DP attention 的 AMD CI(PR-Test(AMD) run 27519156273)据此验证。

## 关联

- #24860(引入问题的 PR)
- PR-Test(AMD) 失败 run:https://github.com/sgl-project/sglang/actions/runs/27519156273

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
