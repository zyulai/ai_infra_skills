# PR #7415: [Kernel] W8A16 Int8 inside FusedMoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mzusman |
| 合并 | 2024-08-16 |
| 分类 | MoE |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/7415 |

## 一句话结论

fused_moe Triton 内核新增 W8A16 Int8 路径(ExpertsInt8),让量化 FusedMoE 首次可跑在 A100,Jamba A100*2 e2e 时延降约 13-26%。

## 动机

此前 FusedMoE 的量化算路只有 FP8 W8A8,依赖 Ampere 之后的 FP8 硬件,A100 等卡上 MoE 只能跑 BF16/慢速 GPTQ(无 fused kernel)。Jamba、Mixtral 等 MoE 模型需要一个不依赖校准数据、质量无损、A100 可用的 MoE 权重量化方案。

## 优化点

`fused_moe_kernel` 增加 `use_int8_w8a16` constexpr 分支:权重以 int8 存储,`tl.dot` 前用 `b.to(compute_type)` 转回 FP16/BF16(W8A16,BF16 计算累加),循环外再乘 per-column per-expert 的 `b_scale`(新增 `stride_bse/bsn`,scale 形状 `[E, N]`,与 FP8 的 per-expert 标量不同);累加器最后 `* b_scale` 反量化。上层新配置 `use_fp8_w8a8`/`use_int8_w8a16` 取代单一 `use_fp8`,config 文件按 dtype_str(`fp8_w8a8`/`int8_w8a16`)命名。新量化方案 `ExpertsInt8Config/ExpertsInt8MoEMethod`:create_weights 分配 int8 权重 + fp32 per-column scale;weight_loader 包装为在线量化——加载 BF16/FP16 权重分片时按行 max-abs/127 对称量化并写 scale,无需校准预处理(Jamba 启动量化约 1 分钟)。另将 Jamba 的 MLP 改为 E=1 的 JambaMoE 复用 FusedMoE 路径,load_weights 把非 MoE 的 `feed_forward` 映射到 `experts.0`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: 内核 W8A16 分支 + use_fp8_w8a8/use_int8_w8a16 参数化
vllm/model_executor/layers/quantization/experts_int8.py: 新增 ExpertsInt8 配置、在线量化 weight_loader
benchmarks/kernels/benchmark_moe.py: 支持 int8_w8a16 的调优/基准
vllm/model_executor/models/jamba.py: MLP 并入 MoE(E=1)路径
```

## 性能声明

E2E 时延(秒,prompt 1024 / decode 128):Mixtral8x22B H100*8:FP8 1.3/1.79/2.26 vs ExpertsInt8 1.3/1.84/2.205(BS=1/4/8);Mixtral8x7B H100*4:FP8 0.835/1.22/1.44 vs Int8 0.83/1.22/1.42;Jamba H100*2:FP8 1.3/2/3.2 vs Int8 1.16/2/3.14;Jamba A100*2:FP16 1.75/3.2/4.6 vs Int8 1.65/2.7/4.0(GPTQ 8bit 无 FusedMoe:3.8/5.9/9.4)。质量(Jamba):gsm8k cot 59.5→59.8、MMLU 67.3→67.3、gsm8k 50.6→50.1、narrative_qa 67.8→68.7、ppl/c4 持平,无退化。

## 验证与回归风险

新增 `tests/quantization/test_experts_int8.py`(Jamba-tiny-random 启动 + 生成冒烟,注释声明"doesn't test correctness");`tests/models/test_jamba.py` 调整且一个用例被 skip。风险面:min capability 80;weight_loader 在线量化改变了 ExpertsInt8 模型的加载语义;FP8 config JSON 文件整体改名(`float8.json` → `fp8_w8a8.json`),依赖配置文件名的工具需同步;`get_config_dtype_str` 影响 MoE 配置缓存键。

## 关联

同区域:FP8 W8A8 FusedMoE(同内核的另一 constexpr 分支,本 PR 将其参数化);后续 Marlin/INT8 MoE 内核线。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
