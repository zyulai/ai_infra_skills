# PR #59044: [Perf][MoE] Skip padded work in SM100 FP4 DeepGEMM experts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tlrmchlsmth |
| 合并 | 2026-10-16 |
| 分类 | MoE |
| Label | performance, ready, quantization |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/59044 |

## 一句话结论

SM100 FP4 DeepGEMM MoE 层跳过 padding 计算(prefix-sum 调度 + activation masking),Full FFN 延迟降 9.36%(1.103x),activation kernel 76.0 → 20.7 µs。

## 动机

Blackwell FP4 DeepGEMM MoE 层目前对 padding 行也在两个 GEMM 和 activation 中做完整计算,主要影响 DeepEPv2 + DeepGEMM 下的 decoder。per-row 布局元数据多达 112 KiB(32 个 local experts、28,672 行分配容量),且 packed SiLU+mul+FP8 量化 kernel 对全部分配行运行。

## 优化点

两处机制改动:(1) 分组 GEMM 改用 DeepGEMM 的 prefix-sum scheduler——每个 expert 的 endpoint 为"对齐的 expert 起点 + 实际 token 数",下一个 expert 从前一 endpoint 向上取整到 GEMM 对齐处开始,无需 count readback、额外 launch 或改 DeepGEMM 源码,per-row 布局元数据从 112 KiB 缩到 128 bytes;(2) packed SiLU+mul+FP8 量化 kernel 增加 expert endpoints 参数,只处理 live rows(padding 输出不写,down GEMM 用同样的 bounds 识别 live 行)。量化元数据是可选的:不传 metadata 的调用保持原 dense launch 配置,metadata 加载被编译掉;SM120 与 FP8 expert 调用方保持原路径;`mk_alignment_scope` 与 `ensure_zero_padding=True` 保留。activation workspace 分配容量不变。与 #51332/#51217/#53911 的区别:本 PR 由 `DeepGemmFP4Experts` 直接把 expert endpoints 传入 packed-UE8M0 fused activation kernel。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py: use_psum_layout(family 100)+ expert_ends/expert_alignment 传入 _act_mul_quant 与两个 GEMM
tests/kernels/moe/test_silu_mul_per_token_group_quant_fp8_colmajor.py: CUDA-graph replay 跳 padding 测试,is_cuda → is_cuda_alike
```

```python
a2q, a2q_scale = self._act_mul_quant(
    input=mm1_out.view(-1, N), output=quant_out, activation=activation,
    expert_ends=grouped_layout if use_psum_layout else None,
    expert_alignment=align_used if use_psum_layout else 0)
```

(fp8_utils 中 packed 量化函数的 expert_ends 参数等实现位于本 pack 未含的文件。)

## 性能声明

验证环境:8x GB200,DeepSeek-V4-Flash(revision 60d8d707),EP8/DP8/TP1,DeepEP V2 + DeepGEMM。成对对比(prefix-sum GEMM 两种变体,均开 prefix-sums):Full FFN 未 profile 425.5 µs → 385.7 µs(-9.36%,1.103x),输出逐 bit 一致;SiLU+mul+quant 单独 profile 76.0 µs → 20.7 µs;standalone 量化(3,072 live / 28,672 allocated rows,hidden 2,048)63.984 µs → 10.510 µs。dense 路径 control:-0.20%(快)到 +0.39%(慢);live 0 行 case 63.986 → 5.254 µs,hidden 7168 的 3,072-live case 217.764 → 27.852 µs。更早的 scheduler-only 验证:stock 717.4 µs vs prefix sums 428.7 µs(同一 FFN 设置)。GSM8K(1,319 题,5-shot,greedy,concurrency 64,FP8 KV):prefix sums only 94.996% vs +masking 95.451%(作者注明这只是 accuracy check,非质量提升证据)。

## 验证与回归风险

GB200 上 69 个 activation 测试通过(含 bitwise live FP8/scale、poisoned alignment gaps、空 expert、全空 counts、graph replay 变 counts);新 graph-replay 测试 guard 从 `is_cuda` 放宽为 `is_cuda_alike`(CUDA/ROCm)。风险:metadata-aware 调用对已满 buffer 有小的 checking/scheduling 开销(28,672x2,048 满载 64.082 → 64.822 µs);SM103 未测(guard 是 SM10.x family);冲突解决后的 revision 未重跑 GPU 测试(本地无 CUDA GPU),FFN/GSM8K 数字来自更早的已验证 revision;结果基于 pinned image 上的源码 overlay,不是全量 main 重建。

## 关联

#51332、#51217(shared masked activation / Humming 路径)、#53911(Humming indexed metadata preparation)——同区域但均不把 expert endpoints 传入该 fused activation kernel。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
