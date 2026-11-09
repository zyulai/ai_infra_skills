# PR #59815: [Quantization] Support compressed-tensors MXFP4 MoE with FP8 activations via Mxfp4MoEMethod

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-10-14 |
| 分类 | MoE(major) |
| Label | ready, deepseek, quantization, DSv4 |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(10 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/59815 |

## 一句话结论

让 compressed-tensors 的 MXFP4 权重 + 动态 FP8 激活 checkpoint 走原生 Mxfp4MoEMethod,Humming MoE 不再退化为 W4A16,专家 GEMM 最高快 1.8×。

## 动机

RedHatAI 的 MXFP4xFP8 系列checkpoint(Qwen3.6-35B-A3B-MXFP4-DSFP8、GLM-5.2/5.3-MXFP4xFP8_BLOCK、转 CT 的 DeepSeek-V4-Flash)此前加载时,Humming MoE 转换只从 `VLLM_HUMMING_INPUT_QUANT_CONFIG` 构建 input schema(代码里留有 TODO),env 未设置就跑 BF16 激活(W4A16);schema fallback 还会把 group-scaled FP8 降级为 A16。输出看起来正确,慢路径因此一直没被发现。

## 优化点

`Mxfp4MoEMethod` 增加可选 `activation_key`,HUMMING 转换用它构造 input schema 且 `allow_input_schema_fallback=False`(env 覆盖优先,其他后端忽略该 key)。新增 `CompressedTensorsW4A8Mxfp4MoEMethod`,继承 `Mxfp4MoEMethod`,只处理 `*_weight_packed` 的 checkpoint 命名(rename/rename-back),后端选择、权重转换、size round-up、monolithic kernel 与 EPLB 全部继承原生 MXFP4 方法。Dense 层用新 `CompressedTensorsW4A8Mxfp4` scheme 请求 FP8 激活(kernel 不支持时回退 W4A4 并告警)。`get_moe_method` 改用层自身的 `ckpt_*_proj_name` 查找 expert scheme,DeepSeek-V4(NVIDIA)传 `ckpt_names=("w1","w2","w3")` 使 `re:.*experts\.\d+\.w\d+$` 类 name-based target 生效。

## 关键改动

```text
vllm/model_executor/layers/quantization/compressed_tensors_moe/compressed_tensors_moe_w4a8_mxfp4.py: 新增 MoE method(继承 Mxfp4MoEMethod)
vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_mxfp4.py: 新增 dense scheme
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py: HUMMING 转换传入 activation_key 且禁用 fallback
vllm/model_executor/kernels/linear/mxfp4/humming.py: dense 层保留 checkpoint 声明的 FP8 激活
vllm/models/deepseek_v4/nvidia/model.py: _init_fused_moe_experts 传 ckpt_names=("w1","w2","w3")
```

## 性能声明

GLM-5.x expert GEMM shapes(N/K 4096/6144 与 6144/2048,32 experts top-8,`benchmarks/bench_humming.py`):W4A8 比 W4A16 快 1.4–1.8×(M≥256)。E2E 反例:小模型 Qwen3.6-35B-A3B 在 1x H200 上 W4A8 反而比 W4A16 慢 3–8%(hidden 2048 / expert intermediate 512,每层激活量化开销摊不掉)。GSM8K 5-shot 1319 题(H200):BF16 0.956 / MoE W4A16 0.958 / MoE W4A8(本 PR)0.961。

## 验证与回归风险

验证:`tests/quantization/test_compressed_tensors.py -k mxfp4` 4 passed(H200,新增 scheme 路由/激活 key/共享原生 method 三个用例);H200 上以 inspecting loaded layers 确认 Humming FP8 GS128 生效(TP1/TP2/TP2+EP,greedy 输出 TP2 与 TP2+EP 一致);B300 上 layer harness 与原生 `Mxfp4Config` 在 7 个后端 bitwise 相等;转 CT 的 DeepSeek-V4-Flash-0731 与原 checkpoint 在 8 prompts 上 token ids/logprobs 一致。风险:依赖 #61556(Humming 0.1.16 拒绝该组合);ROCm/XPU/CPU 无 FP8 激活的 MXFP4 linear kernel(告警回退);Blackwell dense 走 Humming FP8 比 FlashInfer W4A4 慢,MoE Blackwell 后端尚未接 activation key(follow-up);`supports_pre_processed_weights` 关闭;DeepSeek-V4-Flash 转换器的 `shared_experts` 命名问题待修。

## 关联

依赖 #61556;与 #59376 为同一 scheme 的替代设计(二选一);#58714 为同 checkpoint 的独立 CT→HummingMoEMethod 路线(本 PR 覆盖);#56523 / #56306 迁移 W4A4 方法,不涉及本 scheme。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
