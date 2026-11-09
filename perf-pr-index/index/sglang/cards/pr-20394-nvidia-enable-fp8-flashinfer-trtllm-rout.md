# PR #20394: [NVIDIA] Enable fp8 flashinfer_trtllm_routed MoE for MiniMax-M2.5

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @trevor-m |
| 合并 | 2026-04-02 |
| 分类 | MoE(major) |
| Label | documentation, quant, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/20394 |

## 一句话结论

为 MiniMax-M2.5 打通 fp8 flashinfer_trtllm_routed MoE 路径,4×GB200 上 TP4 吞吐较默认 triton MoE 提升 9.04%。

## 动机

MiniMax-M2.5 的 fp8 MoE 之前只能走默认 triton runner;FlashInfer 的 TRTLLM fp8 MoE 内核已有 routed(topk 输出格式)变体,但 SGLang 侧的权重对齐、输出 dtype 与路由方法类型等细节没有对齐该路径,无法在 MiniMax 上启用。

## 优化点

四点适配:内核输出固定 bf16,故 symm_output 改按 `hidden_states.dtype` 分配并在需要时回cast;为 routed 版本启用 `align_fp8_moe_weights_for_flashinfer_trtllm` 的 FP8 权重对齐(fused MoE 加载后处理中把 backend 判断放宽到 `is_flashinfer_trtllm_routed`);routed 与 non-routed 复用同一个 fused func,用 `TopKOutputChecker.format_is_bypassed` 判定走哪条分支;修 `getattr(layer, "routing_method_type", None) or RoutingMethodType.DeepSeekV3`,修复属性被显式设为 None 时 getattr 不返回默认值的问题。另外 autotune 暂不对 routed 后端开启(FlashInfer bug #2749),并留下 TODO:待 #2703 修复后把输出张量直接传给内核、去掉 `symm_output.copy_(output)` 拷贝。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: symm_output 按 hidden dtype 分配;routed 分支加 format_is_bypassed 断言
python/sglang/srt/layers/quantization/fp8.py: FP8 MoE 权重对齐放宽到 trtllm_routed;routing_method_type None 修复
python/sglang/srt/model_executor/model_runner.py: _should_run_flashinfer_autotune 暂排除 flashinfer_trtllm_routed
```

## 性能声明

4×GB200,MiniMaxAI/MiniMax-M2.5,benchmark_serving random(1280 prompts,input 1024/output 1024,range-ratio 0.8,max-concurrency 128,ignore-eos,`--kv-cache-dtype fp8_e4m3`,`--mem-fraction-static 0.9`):TP4 下 flashinfer_trtllm_routed 较默认(triton)MoE 快 9.04%(输出吞吐 3198.31 vs 2933.23 tok/s,总吞吐 6418.44 vs 5886.46 tok/s,duration 368.20 vs 401.48 s);TEP4 下快 5.48%(3215.76 vs 3048.70 tok/s 输出,总 6453.45 vs 6118.18 tok/s)。TPOT:TP4 38.75 vs 42.27 ms,TEP4 38.50 vs 40.61 ms。

## 验证与回归风险

精度:TP4 Accuracy 0.946、Invalid 0.000;TEP4 Accuracy 0.952、Invalid 0.000(未注明具体评测集)。无新增单元测试。风险面:已知 FlashInfer 上游问题限制该路径——#2749 阻止 autotune(已绕过但无额外收益)、#2703 阻止传输出张量(现多一次 copy_);monolithic 内核更快,作者已提 feature request #2736 支持 MiniMax 路由方法。`routing_method_type` 的 getattr 修复影响所有走该 apply 路径的模型(语义上等价恢复默认值)。

## 关联

无 SGLang 侧 PR 编号(外部依赖:flashinfer-ai/flashinfer issues #2749、#2703、feature request #2736)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
