# PR #20339: [TPU][Bugfix] fix the MoE OOM issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20339](https://github.com/vllm-project/vllm/pull/20339) |
| 作者 | yaochengji |
| 合并日期 | 2025-07-06 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | resource |

## 一句话结论
TPU 上 MoE 层不再包装为自定义 op,直接走 `forward_impl`,避免 XLA 下 HBM OOM。

## 问题现象
TPU 上运行 MoE 模型时出现 HBM OOM。作者以 `vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 8 --max-num-batched-tokens 4096 --max-num-seqs 256 --max-model-len 2048 --no-enable-prefix-caching`(等)为测试场景,不加修复即 HBM OOM,修复后通过。影响面:TPU + torch.compile 路径下的 MoE 模型(Mixtral 类)。

## 根因
正文披露:XLA 后端(TPU)自己处理 functionalization,无需把 MoE forward 包成 custom op 来借助 torch.compile 的自动 functionalization;而使用 custom op(`torch.ops.vllm.moe_forward`)会在 TPU 上导致 HBM OOM。OOM 的具体内存机制正文未进一步展开,仅给出上述结论与修复后 OOM 消失的实测。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/layer.py` 中 `FusedMoE.forward`:TPU 平台绕过 custom op,直接调用 `forward_impl`;其余平台保持 custom op 路径不变。代码中留有 TODO:待 TPU 后端的 OOM 问题解决后再切回 `moe_forward` custom op。

```diff
-        return torch.ops.vllm.moe_forward(hidden_states, router_logits,
-                                          self.layer_name)
+        # TODO: Once the OOM issue for the TPU backend is resolved, we will
+        # switch to using the moe_forward custom op.
+        if current_platform.is_tpu():
+            return self.forward_impl(hidden_states, router_logits)
+        else:
+            return torch.ops.vllm.moe_forward(hidden_states, router_logits,
+                                              self.layer_name)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 TPU 平台的 FusedMoE forward 行为改变(放弃 custom op 的编译路径,回到 `forward_impl` 直接执行);非 TPU 平台(CUDA 等)走原 `torch.ops.vllm.moe_forward` 路径,不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
