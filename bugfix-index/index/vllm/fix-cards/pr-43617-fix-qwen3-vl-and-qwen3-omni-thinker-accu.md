# PR #43617: Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43617](https://github.com/vllm-project/vllm/pull/43617) |
| 作者 | andakai |
| 合并日期 | 2026-05-27 |
| labels | ready, qwen |
| 热度 | 👍 3 · 💬 2 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论

Qwen3-VL/omni-thinker 在无视觉时返回 None 的 deepstack 输入使编译图特化到无 deepstack 路径,真实多模态请求复用该图跳过视觉注入;改为始终返回 IntermediateTensors(空路径零填充)。

## 问题现象

在 vLLM 默认 torch.compile / CUDA graph 路径下,Qwen3-VL 出现明显精度劣化:Geo3K 全量评测(601 样本,temperature=0.0,8x H20,基线 commit `d4004455d`)中,Qwen3-VL-2B-Instruct 默认 compile+cudagraph 得分仅 0.2047,而 no compile+no cudagraph 为 0.2912;Qwen3-VL-8B-Instruct 同样 compile 路径 0.5225 低于 no compile 0.5491。问题在纯文本 Qwen3 模型上不复现,指向多模态 deepstack 路径(issue #43602)。

## 根因

正文披露:Qwen3-VL 用 `deepstack_input_embeds` 把视觉特征注入 decoder;修复前 `_get_deepstack_input_embeds()` 在 `deepstack_input_embeds_num_tokens == 0` 时返回 `None`。在 compile warmup / profiling 阶段(无视觉输入)这会使 decoder 编译图特化到「无 deepstack」路径;之后真实多模态请求即使携带真实的 `deepstack_input_embeds` 也可能复用该编译图,导致视觉 deepstack 加法被跳过或处理错误。即 decoder 输入契约不稳定:compile/profiling 路径是 `None`,真实 VL 请求路径是 `IntermediateTensors(...)`。

## 修复方案

修改 `vllm/model_executor/models/qwen3_vl.py` 与 `qwen3_omni_moe_thinker.py` 的 `_get_deepstack_input_embeds()`:删除 `num_tokens == 0` 时返回 None 的分支,只要 deepstack buffer 存在就总是返回 `IntermediateTensors`;对 no-vision/dummy 路径,先用新的 `_resize_deepstack_input_embeds(num_tokens)` 辅助函数把 buffer 重建为零张量(语义等价于无 deepstack 贡献,但保持编译图输入结构不变)。resize 逻辑从 `_set_deepstack_input_embeds()` 抽出复用,保证 get/set 两边一致。关键 diff:

```diff
-        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
-            return None
+        if num_tokens > self.deepstack_input_embeds[0].size(0):
+            self._resize_deepstack_input_embeds(num_tokens)
```

修复后 Qwen3-VL-2B 默认 compile+cudagraph 得分 0.2047 → 0.2829,8B 为 0.5225 → 0.5657(超过 SGLang 参考值 0.5591),推理耗时也下降。

## 影响与回归风险

- 影响窗口:正文给出的基线 commit 为 `d4004455d ([Kernel] Remove NormGateLinear (#43554))`,未说明回归引入的具体版本。
- 回归面:仅涉及 Qwen3-VL 与 Qwen3-omni-moe-thinker 两个模型的 deepstack 输入契约,不改视觉编码器输出、采样配置、chat template 或请求预处理。dummy/无视觉路径现在会分配零张量 buffer,略有额外显存/拷贝开销;若还有其他调用方依赖「num_tokens==0 时返回 None」的旧行为,需要适配。

## 关联

- https://github.com/vllm-project/vllm/issues/43602

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
