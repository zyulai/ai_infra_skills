# PR #51386: [Fix] spec/dflash: Turn the DFLASH draft KV ring off by default

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#51386](https://github.com/sgl-project/sglang/pull/51386) |
| 作者 | Fridge003 |
| 合并日期 | 2026-11-19 |
| labels | documentation, speculative-decoding |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

DFLASH draft KV ring 此前默认开启,而 PD 双端各自按自己的启动参数独立决定是否启用,导致一端开一端关时两端注册的状态条目不一致、decode 调度器在 bootstrap 阶段崩溃;本 PR 把 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING` 默认值改为 `False`(显式 opt-in),并让不支持 ring 的启动在显式开启时于启动期直接报 `ValueError`。

## 问题现象

在 `test_glm53_cp_bcg_dflash.py`(extra-b-test-8-gpu-b300)上,decode 侧报错:

```
RuntimeError: PD state entry mismatch: only the decode server registers a DFLASH draft KV ring
```

具体场景:prefill 侧带 `--attn-cp-size 4 --enable-prefill-cp`,ring 在该端自动关闭("an MHA draft's ring has no context-parallel layout");decode 侧没有 CP,ring 自动开启。异构 TP 的 PD 部署同样以这种方式失败。

## 根因

正文披露:#50740 将 per-request DFLASH draft KV ring 对 all-sliding-window MHA draft 默认打开(叠加 #49664 的 MLA ring)。每个 PD 角色只依据自己的启动参数决定是否使用 ring;当其中一个角色无法服务 ring 而另一个可以时,两个 server 各自注册出不同的状态条目,decode 调度器在 bootstrap 时崩溃。此外,旧逻辑中未显式设置开关时会静默回退到关闭,而只有显式 `=1` 才报错——某一方静默回退正是造成 mismatch 的直接原因。

## 修复方案

1. `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING` 默认值由 `True` 改为 `False`,ring 变为 opt-in(`python/sglang/srt/environ.py`)。
2. `_resolve_dflash_draft_kv_ring`:开关已设置但启动方式无法服务 ring 时,一律在启动期抛 `ValueError`,删除"未显式设置时静默回退"的分支。
3. PD mismatch 报错信息改为提示"在注册 ring 的一端 unset 该开关,或两端都设为 1"(`python/sglang/srt/disaggregation/utils.py`)。
4. 文档同步:更新 `speculative_dflash_draft_kv_ring` 帮助文本、Kimi-K3.1-Pro cookbook 与 mamba-ratio 计算器(计算器仅在 `=1` 时计入 ring);cookbook 的 DFLASH 选项在所有 NVIDIA 单元(含两个 PD 角色)显式设置 `=1`;MI350X/MI355X 用 Triton draft backend(无 ring)保持不变。
5. 单元测试:新增"默认关闭"用例,并把"不支持的启动/draft"用例改为期望显式 opt-in 下的启动期报错。

关键 diff 摘录:

```diff
-    SGLANG_ENABLE_DFLASH_DRAFT_KV_RING = EnvBool(True)
+    SGLANG_ENABLE_DFLASH_DRAFT_KV_RING = EnvBool(False)
```

```diff
     if reason is not None:
-        if switch.is_set():
-            raise ValueError(f"SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1, but {reason}.")
-        logger.info("DFLASH draft KV ring: off, %s.", reason)
-        return
+        raise ValueError(f"SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1, but {reason}.")
```

## 影响与回归风险

- 行为变化:MLA ring(Kimi-K3.1 DFLASH 在 `trtllm_mla` / `cutedsl_mla` 上)现在默认同样关闭。想要约 15% 额外 target KV 的部署需要在两个 PD 角色上都设置 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1`。
- 生成的 AgentX snippet(`docs/src/snippets/agentx/...`)刻意不改(由 InferenceX 同步并自带镜像);Muse-Spark-Open(MHA ring)保持默认关闭。
- 回归面:若已有部署依赖"默认开启"获得 KV 收益,升级后 ring 会静默关闭,吞吐/显存特性回退但不报错;显式 `=1` 但环境不满足时会在启动期立即报错(而不是静默降级),这属于有意为之的快速失败。正文注明本地无 torch 环境,依赖 CI 跑 `test_dflash_draft_kv_ring.py` 与 `test_unified_window_wire.py`,`/rerun-test` 尚未执行。

## 关联

- 前置 PR:#50740(默认开启 MHA ring)、#49664(MLA ring)。
- CI run:[extra-b-test-8-gpu-b300 失败记录](https://github.com/sgl-project/sglang/actions/runs/44196383565/job/134439923646)。
- 涉及测试:`test/registered/unit/spec/test_dflash_draft_kv_ring.py`、`test/registered/unit/disaggregation/test_unified_window_wire.py`、`test_glm53_cp_bcg_dflash.py`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
