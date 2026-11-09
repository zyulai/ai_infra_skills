# PR #8651: [Core] Default to using per_token quantization for fp8 when cutlass is supported.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elfiegg |
| 合并 | 2025-01-16 |
| 分类 | Quantization（tier: major） |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/8651 |

## 一句话结论

cutlass 支持时 fp8 默认改用 per_token 量化以融合反量化，Llama2-7B/H100 吞吐 +4.68%。

## 动机

PR 正文说明的目标是：在 cutlass 受支持时默认启用 fp8 的 per_token quantization，使 dequantization 能与 cutlass GEMM kernel 融合，并通过融合 amax + quantization 改善性能。正文未进一步展开改动前的具体瓶颈、触发场景或更详细的动机，PR 未注明。

## 优化点

改动集中在 `vllm/model_executor/layers/quantization/fp8.py`：线性层 `apply` 调用把 `use_per_token_if_dynamic` 从写死的 `False` 改为 `self.cutlass_fp8_supported`。即当 cutlass fp8 可用时改走 per_token（dynamic）量化路径，让反量化融合进 cutlass GEMM kernel；cutlass 不可用时行为保持不变。正文称由此启用 amax + quantization 的融合。从 diff 看，该调用位于 `apply` 方法内，同处还传入 `input_scale=layer.input_scale` 与 `bias=bias`，本次仅替换末一个实参；紧随其后的 `Fp8MoEMethod` 定义本次未改动。

## 关键改动

```text
vllm/model_executor/layers/quantization/fp8.py: use_per_token_if_dynamic 由 False 改为 self.cutlass_fp8_supported
（本 pack 的 diff 仅含此 1 个文件，无法列出更多核心文件）
```

```python
             cutlass_fp8_supported=self.cutlass_fp8_supported,
-            use_per_token_if_dynamic=False)
+            # Default to using per_token quantization if cutlass is supported
+            use_per_token_if_dynamic=self.cutlass_fp8_supported)
```

## 性能声明

Tested with Llama2-7B on H100：Before `Throughput: 152.60 requests/s, 10681.75 tokens/s`；After `Throughput: 159.73 requests/s, 11181.40 tokens/s`；**About 4.68% speedup**；`Accuracy remains similar if not slightly higher`。测试条件：模型 Llama2-7B、硬件 H100；PR 未注明精度与 workload 细节。

## 验证与回归风险

PR 正文未提及引入的测试或 CI 验证，仅给出 H100 上前/后吞吐对比与 accuracy 的定性说明（"similar if not slightly higher"，无定量指标）。风险面：行为变更仅在 `self.cutlass_fp8_supported` 为真时生效，即支持 cutlass fp8 的路径；所有经该 `apply` 的 fp8 线性层在 cutlass 可用时都会改走 per_token 量化，未给出其他硬件/精度下的验证结果。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
