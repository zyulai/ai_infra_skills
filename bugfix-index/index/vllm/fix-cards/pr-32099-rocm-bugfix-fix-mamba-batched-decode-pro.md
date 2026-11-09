# PR #32099: [ROCm][Bugfix] Fix Mamba batched decode producing incorrect output

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32099](https://github.com/vllm-project/vllm/pull/32099) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-01-13 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 Mamba 系模型 batch>1 decode 输出乱码:`causal_conv1d_update` 返回非连续 view,送入 GEMM 线性层时除最后一批位置外全算错;在进线性层前按 ROCm 条件做 `.contiguous()`。

## 问题现象
ROCm 上 Mamba 系模型(Jamba、Mamba、FalconMamba、PLaMo2 等)在多个序列一起 decode(batch size > 1)时输出不一致且错误;单序列推理正常。正文示例:单 prompt「The capital of France is」正确生成「 Paris, a city of 12 million...」;同 prompt 的 batch of 3 则三条各自出现「 Paris, a city of culture, history...」「 Paris, or circ, for you conveni...」等错误文本。

## 根因
正文披露:`causal_conv1d_update` 返回非连续的 tensor view。该 tensor 经 transpose 后传入线性层(MambaMixer 的 `x_proj`、PLaMo2 的 `bcdt_proj`)时,ROCm 的 GEMM kernel 对除最后一个 batch 位置以外的位置算出错误结果。正文附调试证据:位置 1 与 2 的线性层输入完全相同(`[-0.048828125, -0.1259765625, ...]`),输出却不同(`B_d[1]` 的 `-1.3125, 8.9375` 为 WRONG,`B_d[2]` 的 `-0.828125, 4.0` 正确);`hidden_states_d.is_contiguous(): False`,`stride(): (8192, 1)`(期望 `(4096, 1)`,shape [batch, 4096])。

## 修复方案
在传入线性层前确保 tensor 连续,均以 `current_platform.is_rocm()`(或 LoRA 条件)守卫,避免 CUDA 等平台产生多余拷贝:
1. `vllm/model_executor/layers/mamba/mamba_mixer.py` `_ssm_transform()`:把原有的仅 LoRA 分支改为 `if self.is_lora_enabled or current_platform.is_rocm(): x = x.contiguous()`,再统一 `self.x_proj(x)`;
2. `vllm/model_executor/models/plamo2.py` decode 路径:`causal_conv1d_update` 之后、`_project_ssm_parameters`(`bcdt_proj`)之前,ROCm 上执行 `hidden_states_d = hidden_states_d.contiguous()`。

```diff
-        if self.is_lora_enabled:
-            ssm_params = self.x_proj(x.contiguous())[0]
-        else:
-            ssm_params = self.x_proj(x)[0]
+        if self.is_lora_enabled or current_platform.is_rocm():
+            x = x.contiguous()
+        ssm_params = self.x_proj(x)[0]
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:ROCm 上所有使用 `MambaMixer` 的 Mamba 系模型(Jamba、Mamba、FalconMamba、PLaMo2)的 batched decode 路径,新增一次 contiguity 拷贝(仅 ROCm);CUDA/其他平台行为不变(LoRA 分支逻辑保持等价)。修复前测试 3 个全失败,修复后 3 个全通过(test_hybrid.py 的 test_models/test_batching/test_full_cuda_graph)。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
