# PR #64759: [Bugfix][Kimi-K3] Pin FlashKDA and FlashMLA with NaN fixes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64759](https://github.com/vllm-project/vllm/pull/64759) |
| 作者 | simon-veitner-redhat |
| 合并日期 | 2026-11-02 |
| labels | bug, ci/build, k3, kimi, ready |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Kimi-K3 高负载下部分请求变成 token 0 流;FlashKDA prefill 越界行 `0*NaN=NaN` 传染同批请求、FlashMLA PDL 缺 wait 读未写 query,移动两个 pin 修复。

## 问题现象
issue #64758 报告:负载下 Kimi-K3 部分请求从某点起持续输出 token 0(`!!!!`)直到 `max_tokens`。验证环境:`moonshotai/Kimi-K3` @ `f831ab66`,2 x 8 H200,TP8 x PP2 over InfiniBand,V2 model runner,vLLM `063c0c6`(PR #61867 head)。自然 burst 复现器(repro_natural_burst.py)上:stock 出现 2 次 first-NaN 事件、14 个坏请求(分 2 和 13 两组);仅 FlashKDA fix 后 1 次/1 个;两个 fix 后 0 次/0 个(预期约 2.6,P≈0.08)。注入一个 NaN 请求时,stock 使同批 240 个其他 decode 中 96 个变坏,FlashKDA fix 后为 0。GSM8K stock 0.945 / 修复后 0.947。

## 根因
PR 正文披露两个根因:

1. **FlashKDA(已确认,spread)**:prefill 按 16 行分块处理,序列最后一个不完整 chunk 会读到序列结束之后的 `beta` 和 `v`(同 batch 的下一请求或下一 head 的行),原实现靠乘 0 抵消这些行,但 `0 * NaN` 仍是 NaN。vLLM 在混合 step 中让 decode 行与 prefill 一起走这个 kernel,因此一个 NaN 请求每步可波及最多 15 个 decode 和 1 个 prefill。kernel 修复改为对这些行直接选 0。
2. **FlashMLA(强嫌疑,未证实)**:K3 decode 路径先跑 `fused_kimi_k3_mla_decode_q_concat_kv_cache_insert`(写 query 并触发 PDL 完成),再跑 FlashMLA 的 SM90 dense decode kernel;该 kernel 以 PDL 启动却从不调用 `griddepcontrol.wait`,可能读到尚未写入的 query 缓冲(`torch.empty`)。修复在 prologue 后加 wait。正文明确说明:未证明该 race 造成自然事件——有一次自然事件发生在 attention 输出之后,该 race 解释不了。

## 修复方案
vLLM 代码本身不变,仅移动 `cmake/external_projects/` 下两个 pin 到包含 kernel 修复的 upstream merge commit:

```diff
-    GIT_TAG 47fc9fec6661d04aa9a42e9e5e069a51ad3b649b
+    GIT_TAG 9d9e928edc42475a67a523dd37814540d105f51c
```

```diff
-        GIT_TAG d1a1c92eff2cf558ddf49293828cd2bd21ca48c4
+        GIT_TAG 847ef308bb7a6828c8117301f5a5a84a095a90e8
```

- FlashKDA:`47fc9fe` → `9d9e928`(merge of vllm-project/FlashKDA#16),无其他上游变更带入。
- FlashMLA:`d1a1c92` → `847ef30`(merge of vllm-project/FlashMLA#41),同时带入 vllm-project/FlashMLA#39(SM100 sparse decode head64 kernel 的每 kernel 一行修复)。

## 影响与回归风险
- 影响窗口:正文未给出 vLLM 版本范围;涉及的是当前 FlashKDA pin(`47fc9fe`)与 FlashMLA pin(`d1a1c92`)到新 pin 之间的外部库状态。
- 回归面:FlashMLA 的 SM90 dense decode kernel 修复影响所有使用该 kernel 的模型(不只 Kimi-K3),每个 CTA 增加一次 wait,输出 bit 级一致;FlashKDA pin 移动无额外上游变更。性能 A/B(3 次中位数):TPOT batch 1/8 基本持平(-0.1%/-0.2%),batch 32/64 +1.8%/+2.1%(在噪声内,3 次重复不能排除 batch 32-64 有 1-2% 代价),TTFT 与 GSM8K 持平。CI 无法运行 Kimi-K3(需两个 8 x H200 节点),上述验证在 CI 外完成;`griddepcontrol.wait` 变体未在自然 burst 复现器上验证(仅强制 kernel race 49/50→0/50 与性能 A/B)。

## 关联
- https://github.com/vllm-project/vllm/issues/64758
- https://github.com/vllm-project/FlashKDA/pull/16
- https://github.com/vllm-project/FlashMLA/pull/41
- https://github.com/vllm-project/FlashMLA/pull/39

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
