# PR #57549: [EPLB] Add EPLB support to dspark draft models (DeepSeekV4)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2026-10-12 |
| 分类 | Speculative Decoding \| tier: major |
| Label | speculative-decoding, ready, deepseek, mrv2, DSv4, dflash, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 1 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/57549 |

## 一句话结论

让 DSpark MoE draft 接入 EPLB，DeepSeek V4 下 target 与 draft 专家可一并重平衡。

## 动机

PR 之前，使用 DSpark speculative decoding 时只有 target 模型参与 EPLB 专家负载均衡，draft 侧专家分布不随 `--enable-eplb` 调整。正文指出 DSpark MoE drafters 现已实现 `MixtureOfExperts`，因而具备让 EPLB 注册并重平衡 draft 专家的条件；同时缺少区分 target 与 draft 的 EPLB 日志，运行过程难以分辨两类模型的重排。

## 优化点

让 `DSparkDeepseekV4ForCausalLM` 继承 `DeepseekV4MixtureOfExperts`，构造时调用 `set_moe_parameters()` / `collect_moe_layers` 收集 MoE 层，使 EPLB 控制器能把 draft 当作 `MixtureOfExperts` 注册；在 expert 权重映射中补上 `num_redundant_experts`，让冗余专家进入映射。v2 GPU model runner 的加载路径接入 speculator EPLB，`eplb_state` 增加 draft 模型日志。正文明确 DSv4.1 DSpark draft 不在支持范围，因为 EPLB 要求 target 与 draft 的 routed experts 数量相同；Qwen3、Gemma4、K3 MLA 等 dense draft 无 MoE 层，draft 侧 EPLB 为 N/A；AMD/XPU 上的 DSV4 DSpark 因 target 路径未暴露 `MixtureOfExperts` 同样不支持。

## 关键改动

```text
vllm/models/deepseek_v4/nvidia/dspark.py: draft 类继承 MixtureOfExperts，收集 moe_layers，权重映射带 num_redundant_experts
vllm/models/deepseek_v4/common/eplb_util.py: 提供 collect_moe_layers 供 draft 复用
vllm/distributed/eplb/eplb_state.py: 注册 draft 模型并增加区分 target/draft 的日志
vllm/v1/worker/gpu_model_runner.py、vllm/v1/worker/gpu/eplb_utils.py: v2 model runner 加载时把 speculator 接入 EPLB
```

```diff
-class DSparkDeepseekV4ForCausalLM(nn.Module):
+class DSparkDeepseekV4ForCausalLM(nn.Module, DeepseekV4MixtureOfExperts):
+        self.set_moe_parameters()
+
+    def set_moe_parameters(self) -> None:
+        collect_moe_layers(
+            self,
+            self.model.layers,
+            self.config,
+            decoder_layer_type=DeepseekV4DecoderLayer,
+            moe_type=DeepseekV4MoE,
+        )
```

## 性能声明

PR 正文未给出改动前后对比式性能声明，仅在 Validation 中给出单次端到端 GSM8K 结果：Accuracy 0.952、Invalid responses 0.000、Total latency 30.602 s、Questions per second 43.102、Total output tokens 118709、Output tokens per second 3879.131。测试条件：客户端 `tests/evals/gsm8k/gsm8k_eval.py`，1319 questions、5-shot；服务端 `deepseek-ai/DeepSeek-V4-Flash-DSpark`，`--tensor-parallel-size 4`、`--enable-eplb`、`--eplb-config.num_redundant_experts=4`、`--kv-cache-dtype fp8`、`num_speculative_tokens=7`；硬件型号 PR 未注明测试条件。日志另给出 EPLB step 0 的 balancedness：draft 0.9437、target 0.8741，avg_tokens 12600.00 / 414348.00。以上为功能验证数字，非 before/after 性能对比。

## 验证与回归风险

Test Plan 列出三个测试：`test_dspark_draft_registers_mixture_of_experts`（draft 暴露 `MixtureOfExperts`）、`test_eplb_registers_dspark_draft_model` / `test_eplb_skips_dspark_registration_with_dummy_weights`（EPLB controller 注册 DSpark draft）、`test_v2_load_model_registers_dspark_speculator_with_eplb`（v2 GPU model runner 加载时接线）；diff 另含 `test_deepseek_v4_mega_moe.py`、`test_dspark_mla.py`、`test_dspark_utils.py`、`test_gpu_model_runner_v2_eplb.py`、`test_eplb_spec_decode.py` 等测试改动。Validation 用真实 server + GSM8K client 确认两个模型都发生 EPLB 重排。回归风险：正文明确 DSv4.1 DSpark draft、dense draft 与 AMD/XPU 路径不支持；改动集中在 EPLB 注册与 v2 model runner 加载，可能影响其它 speculative decoding + EPLB 组合，PR 未注明更多回归路径。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
