# PR #31772: [NPU] Fix LLaDA2 MoE OOM after the FRACTAL_NZ cast, re-enabling the NZ speedup

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31772](https://github.com/sgl-project/sglang/pull/31772) |
| 作者 | wenxuewuhd |
| 合并日期 | 2026-07-21 |
| labels | npu, run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | MoE |
| bug 类型 | resource |

## 一句话结论
load_weights 提前快照 x.data 钉住 ND 旧存储,NZ cast 后 MoE 权重双份驻留致 OOM;改 LazyValue 惰性建图修复。

## 问题现象
NPU 上启用 FRACTAL_NZ 权重 cast 加载 LLaDA2 时,MoE 权重内存约翻倍,64 GB 910B3 上 server 完全无法启动。TP1 / 910B3、CI 测试配置(`--mem-fraction-static 0.9`、NZ cast 开):修复前 weights 59.03 GB、load 73.9 s,报 `ValueError: loaded weights leave no GPU memory for the KV cache`;修复后 weights 30.45 GB、load 38.4 s,KV cache 628736 tokens,server 正常启动。LLaDA2-mini CI 测试此前一直用 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1` 掩盖症状,同时放弃了 NZ kernel 的加速。

## 根因
正文披露:`LLaDA2MoeModelLM.load_weights` 结尾通过 `get_moe_weights()` 构建 `routed_experts_weights_of_layer`,该函数对每个专家权重快照 `x.data`——一个与 Parameter 共享 storage 的独立 Tensor 对象;快照发生在 `process_weights_after_loading` 之前,当 NZ cast 把 `weight.data` 重绑到新分配的 NZ 张量时,map 仍持有 ND 旧 storage 的句柄,引用计数永不归零,每份 MoE 权重同时以两种 layout 驻留。LLaDA2.0-mini 有 19 个 MoE 层、每层 w13 1 GiB + w2 0.5 GiB,共多钉住 28.5 GiB。作者在真实模型加载时遍历 referrer graph 验证:cast 后恰有一个存活张量仍共享 pre-cast storage(同 shape、非 Parameter、可从 dict 键 `routed_experts_weights_of_layer` 到达)。

## 修复方案
1. `python/sglang/srt/models/llada2.py`:把 `routed_experts_weights_of_layer` 包进 `LazyValue`,推迟到 EPLB 真正访问时(权重处理之后)才构建;dict 推导与过滤条件不变,仅移动求值时机(与 `qwen3_moe.py`/`sdar_moe.py`/`deepseek_v2.py` 的现有模式一致):

```diff
+        if not hasattr(self, "routed_experts_weights_of_layer"):
+            self.routed_experts_weights_of_layer = LazyValue(
+                lambda: {
+                    layer_id: layer.mlp.get_moe_weights()
+                    for layer_id, layer in enumerate(self.model.layers)
+                    if not isinstance(layer, PPMissingLayer)
+                    and isinstance(layer.mlp, LLaDA2MoeSparseMoeBlock)
+                }
+            )
```

2. CI 测试 `test_npu_llada2_mini.py` 删除 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1` kill switch,重新启用 NZ cast;整个 `env` 覆盖一并移除(它原本整体替换了 `GSM8KAscendMixin.env`,属顺手修复的无关 papercut,作者确认并非 OOM 成因)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:llada2 的 EPLB 专家权重 map 改为惰性构建,别名指向处理后的存活权重;顺带修复一个非 NPU 特有的潜在正确性 bug——任何 `process_weights_after_loading` 重绑 `weight.data` 的后端,eager 建图会让 EPLB 专家搬移写进死张量。NZ cast 重新启用后 kernel 收益:MoE GroupedMatmul 8305.9→7697.8 us(1.08x,38/38 calls faster)、whole-step 13308.2→12736.0 us(1.045x),dispatch 到 `aclnnGroupedMatmulWeightNz` 且无 TransData 转换开销;gsm8k 首批 200 题分数 0.890 不变,吞吐 242.947→262.055 token/s。作者注明 `bailing_moe.py` 与 `sarvam_moe.py` 仍 eager 构建、存在同样潜在问题,本 PR 未改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
