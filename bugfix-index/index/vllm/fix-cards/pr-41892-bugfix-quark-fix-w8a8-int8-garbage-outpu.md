# PR #41892: [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41892](https://github.com/vllm-project/vllm/pull/41892) |
| 作者 | JoursBleu |
| 合并日期 | 2026-05-13 |
| labels | bug, ready, rocm |
| 热度 | 👍 1 · 💬 3 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
Quark INT8 加载路径三处 bug 导致 Step-3.5-Flash 输出乱码/NaN;补 packed_modules_mapping、重写 MoE per-channel 路径并对齐 2D scale 修复。

## 问题现象
将 `stepfun-ai/Step-3.5-Flash` 用 AMD Quark 量化为 W8A8 INT8 后,经 vLLM 直接 serving 输出乱码(随机 token、频繁 NaN logits)。同一份权重离线改写成 compressed-tensors 格式后用原版 vLLM 服务则完全正常——证明 INT8 数值本身正确,bug 在 vLLM 的 Quark 加载路径。影响面:使用 Quark 导出、`exclude` 中以非融合形式列出 shared expert 的 3-key fused-MoE 模型(Step-3.5-Flash 及类似导出)。修复后 Quark 原生 checkpoint `stepfun-ai/Step-3.5-Flash-Quark-W8A8-INT8` 无需改写直接运行,MI355X TP=2+EP=2 上 smoke 提问从 garbage 恢复为正确回答("The product of 12 and 17 is **204**."),GSM8K(1319 题,8-shot)1255/1319 = 95.15%,与 CT 格式基线 95.07% 持平,对 BF16 基线 95.91% 无损(差距在噪声内)。测试版本:vLLM `0.19.2rc1.dev109+gaad88f848.rocm721`。

## 根因
PR 正文披露三个独立根因,共同构成乱码输出的根因:

1. `Step3p5ForCausalLM` 缺少 `packed_modules_mapping`。`QuarkConfig.get_quant_method` 用它把融合前缀(`qkv_proj`、`gate_up_proj`)反向映射回 `quantization_config.exclude` 中的非融合名。缺失时(默认空 dict),Quark 导出中按非融合形式列出的 shared expert(`model.layers.X.share_expert.{gate,up,down}_proj`)无法匹配,`share_expert.gate_up_proj` 被错误挂成 Quark INT8 linear;checkpoint 里没有它的 INT8 权重与 scale,`weight_scale` 保持零初始化,shared-expert 输出恒为 0/NaN,经残差与 layernorm 传播毁掉整个前向。
2. `QuarkW8A8MoEMethod` 不处理 Quark 的 INT8 checkpoint 布局:Quark 导出带冗余对称 `weight_zero_point`/`input_zero_point` 张量,且 per-channel `weight_scale` 存为 1D `[N]`(无尾随单例维)。旧代码只能凑合跑 legacy per-tensor INT8 配方,对 Step-3.5 用的 per-channel + 动态 per-token 激活配方产生错误数值。
3. 非 MoE 的 `QuarkW8A8Int8` linear 也缺 2D scale 对齐:Quark 把 per-channel `weight_scale` 存成 1D `[N]`,而 vLLM 其余部分与 compressed-tensors 期望 2D `[N, 1]`。

## 修复方案
三处修复,改动仅限 Quark INT8 路径与 `Step3p5ForCausalLM`,compressed-tensors 路径不动:

1. `vllm/model_executor/models/step3p5.py`:给 `Step3p5ForCausalLM` 补上标准 `packed_modules_mapping`(`qkv_proj: [q,k,v_proj]`、`gate_up_proj: [gate,up_proj]`),使 `should_ignore_layer` 正确匹配 exclude 列表,shared expert 回退 `UnquantizedLinearMethod` 正常加载。
2. `quark/quark_moe.py`:重写 `QuarkW8A8Int8MoEMethod`,镜像 `CompressedTensorsW8A8Int8MoEMethod`——per-channel scale 注册为 2D `[E, N]` 加载后在 `process_weights_after_loading` unsqueeze 为 `[E, N, 1]`,kernel 经 `int8_w8a8_moe_quant_config(..., per_act_token_quant=True)` 调用:

```diff
+        # Per-channel scales: 2D [E, N] -> 3D [E, N, 1] for the int8 MoE kernel.
+        if self.weight_qscheme == "per_channel":
+            for attr in ("w13_weight_scale", "w2_weight_scale"):
```

对称 `*_zero_point` 注册为占位参数以通过 safetensors 加载,构建 quant config 前丢弃;legacy per-tensor 静态 W8A8 路径完整保留(shard 合并逻辑不变)。
3. `quark/schemes/quark_w8a8_int8.py`:新增 weight-loader shim,在标准形状检查前把 1D scale unsqueeze 为 `[N, 1]`,参数本身按 `[N, 1]` 分配,运行时布局与 compressed-tensors 一致;per-tensor 标量 scale 不受影响,向后兼容。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是 Quark INT8(per-channel/per-token)导出的模型 serving。
- 回归面:`Step3p5ForCausalLM` 的 `packed_modules_mapping` 与其他 QKV+gate_up 模型一致,只影响 Quark/CT exclude 列表对融合前缀的匹配,不影响权重加载本身;`QuarkW8A8Int8` linear 改动向后兼容(旧 1D scale 加载时重塑);MoE 重写保留 legacy per-tensor 路径,只修 per-channel + 动态输入路径。其他模型与其他量化路径不受影响。

## 关联
- `stepfun-ai/Step-3.5-Flash-Quark-W8A8-INT8`(暴露此 bug 的公开 checkpoint)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
