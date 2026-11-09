# PR #37243: fix: resolve remaining GLM-5.3 rebase regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37243](https://github.com/sgl-project/sglang/pull/37243) |
| 作者 | Fridge003 |
| 合并日期 | 2026-08-31 |
| labels | jit-kernel |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
清理 #37235 合入 #36507 后遗留的 5 组 rebase 回归:Helion KDA 补 `beta_is_raw` 处理、GLM5 混合配置探测容错、DSA CP 测试更新、adaptive-spec 默认值对齐、Frozen-KV MTP 恢复 plan-stream handoff。

## 问题现象
#37235 合入 #36507 后出现确定性的 CI 回归(正文 Summary 逐条列出):Helion KDA prefill 的 `beta_is_raw` 语义未生效;GLM5 混合配置探测对最小化/非 GLM fixture 崩溃;DSA CP 单测引用了已改名的私有 helper;adaptive-spec 测试与 rebase 后的 GLM-5.3 默认值不一致;Frozen-KV MTP 缺少 plan-stream handoff 初始化与 verify 事件发布。验证方式为专用 devbox 与 `/rerun-test`(未跑全量 CI),H100 RX 上:`test_kda_helion.py` 26 passed、Frozen-KV MTP 两种 top-k 各 200 条 GSM8K 通过、gemma4 MTP e2e top-k 1/3 超过 0.75 阈值、adaptive-spec 等回归集 37 tests + 12 subtests 通过、Kimi-K3 前置 kernel 12 passed。

## 根因
正文披露五条根因:
1. Helion KDA 只通过 `**kwargs` 接收 `beta_is_raw`,静默忽略该参数,raw-beta 契约(对 beta 做 sigmoid 转换)没有实现;
2. `glm5_next_config` 假定每个模型/配置 fixture 都暴露 `model_type` 与 `is_draft_model`,最小化或非 GLM fixture 上即崩;
3. 测试保留了 rebase 前的私有 helper 名与 GLM-5.3 adaptive 默认值;
4. plan-stream handoff 支持只在常规 EAGLE worker 中初始化事件状态;Frozen-KV MTP 继承的代码会消费 `_pending_verify_handoff_event`,却从未创建或发布它;
5. 首个 raw-beta 测试 fixture 使用了数学上发散(expansive)的未归一化递推,把普通 BF16 舍入差异指数级放大,造成假失败。

## 修复方案
- `kernels/ops/attention/helion/kda_prefill.py`:`chunk_kda` 增加显式 `beta_is_raw: bool = False` 参数,为真时 `beta = beta.float().sigmoid()`;
- `configs/hybrid_arch.py`:`glm5_next_config` 改用 `getattr(hf_config, "model_type", None)` 与 `getattr(model_config, "is_draft_model", False)` 容错;
- `speculative/base_spec_worker.py`:把 handoff 事件初始化抽为 `_init_handoff_events()`(plan-stream、每类 8 个事件的环形缓冲、`_pending_verify_handoff_event=None`)与 `_record_handoff_event()`;`eagle_worker_v2.py` 改为调用之,`frozen_kv_mtp_worker_v2.py` 在 `__init__` 中补调 `self._init_handoff_events()`,并在 `draft()` 构造 `FrozenKVMTPVerifyInput` 后记录 verify handoff 事件:
```diff
+        self._pending_verify_handoff_event = self._record_handoff_event(
+            "verify", torch.get_device_module(self.device).current_stream()
+        )
+        return verify_input
```
- 测试:`test_dsa_trtllm_fp8_cp.py` 换用 `_should_return_dsa_dcp_lse` 并改继承 `CustomTestCase`;`test_adaptive_spec_params.py` 对齐新默认(bs=1 slot 候选步 `[1,3,5,7]`、`resolve_candidate_steps` 为 `[0,1,3,5,7]` 等);`test_kda_helion.py` 新增 `test_raw_beta_prefill_contract`(收缩性 keys ×0.05、衰减 gate `-rand*0.2`、raw beta 均布于 [-2,2])。

## 影响与回归风险
- 影响窗口:正文未说明(范围为 #37235 合入后的回归)。
- 回归面:`chunk_kda` 新增显式参数对既有调用默认不变;`glm5_next_config` 的 getattr 容错只放宽判定、不改 GLM 路径行为;handoff 初始化逻辑从 `eagle_worker_v2` 移到基类,Frozen-KV MTP 新增事件记录使 plan-stream GPU 工作与 forward 流正确同步(行为对齐常规 EAGLE worker);其余均为测试文件修正,不影响运行时。

## 关联
- sgl-project/sglang#37235(引入回归的 PR)
- sgl-project/sglang#36507(合入目标)
- sgl-project/sglang#37250(承接了本 PR 原含的 fused sigmoid-gating split-grid 修复)
- https://github.com/sgl-project/sglang/actions/runs/33426961755 (lint)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
