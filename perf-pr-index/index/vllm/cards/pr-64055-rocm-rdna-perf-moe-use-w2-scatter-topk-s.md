# PR #64055: [ROCm][RDNA][Perf][MoE] Use w2 scatter + topk sum for the tk 3.8 OAI MoE combine on RDNA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @umarkovi-amd |
| 合并 | 2026-11-12 |
| 分类 | MoE(tier: major) |
| Label | rocm, ready, gpt-oss |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64055 |

## 一句话结论

RDNA 上把 triton_kernels 3.8 OAI Triton MoE 的输出 combine 改回 v0.29 `matmul_ogs` 的方式——w2 `matmul` 用 `scatter_indx` 直接 scatter 到 token 序,`masked_moe_sum` 加 top-k rows,替代 fp32 `zeros` + `index_add_` + cast(在 gfx1201 上占 gpt-oss-20b GPU 时间约 20%):R9700 吞吐 +17.6%,输出与 `index_add_` 路径 bitwise 一致。

## 动机

#55934 把 OAI Triton MoE 移到 triton_kernels 3.8:其 `matmul` 丢掉了 `matmul_ogs` 的 in-kernel top-k reduce,但保留了 in-kernel row scatter(`scatter_indx`,见 `matmul_details/_matmul.py`);移植时传 `scatter_indx=None`,改用 atomic fp32 `index_add_` combine。gfx1201 上该 `indexFuncLargeIndex<float, …, ReduceAdd>` kernel 每次调用约 590 µs(约 1.8 µs/token),占 forward 的 19%——main 因此比 v0.29 低 13.5%。

## 优化点

- RDNA 上,`RoutingData`(3.8)新增携带 `scatter_indx`(既有的 sorted-row → `token * topk + slot` 序,int32,否则 `None`);w2 `matmul` 带上它,使 `y` 直接以 `(M * topk, K)` token 序输出,`masked_moe_sum`(`topk_ids=None`,新增语义)在 fp32 中对每个 token 的 top-k rows 求和并写 bf16——同样的算术,输出 bitwise 一致。
- Expert-map 的 −1(non-local)slots 保持精确零:epilogue 计算 `(acc + bias) * gamma`,gamma 为 0。
- Split-K 不受影响:3.8 对 ragged MXFP4(`is_ragged_mx`)本就禁用 split-K,与 `matmul_ogs`(`not w_has_mx`)一致。
- 新路径(含 scatter index 构建)由 `on_rdna()` 门控,RDNA 之外零变化;`_TK38_SCATTER_COMBINE = current_platform.is_rocm() and on_rdna()`。
- 已知限制:w2 `matmul` 的 scatter 变体是新的 Triton specialization,JIT monitor 首次使用会 log 一条 `define_kernel.<locals>_empty_fn` 编译消息(与 baseline 的 `_topk_forward`/bitmatrix kernels 同类);模块化/专家并行类 `OAITritonExperts` 仍用 `index_add_`;CDNA 与 NVIDIA 可能同样受益,但 scatter 改变 w2 store 路径(如 Hopper 无 TMA store),解除门控需相应 owner 验证。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py: _TK38_SCATTER_COMBINE 门控;RoutingData 增加 scatter_indx;triton_kernel_fused_experts 的 3.8 路径把 scatter_indx 传给 w2 并在 RDNA 上用 masked_moe_sum;masked_moe_sum 支持 topk_ids=None(HAS_TOPK_IDS constexpr)
tests/kernels/moe/test_gpt_oss_triton_kernels.py: scatter combine 与 index_add 对照测试(含 expert map -1 slots)与 topk_ids=None 求和测试
```

```python
     if triton_kernels_version == "3.8":
-        # 3.8 has no in-kernel scatter+topk-reduce, so the two matmuls are
-        # followed by an external index_add combine.
+        # 3.8 has no in-kernel topk-reduce, so the two matmuls are followed by
+        # an external combine.
         assert quant_config is not None
         M, K = hidden_states.shape[-2:]
@@ -761,13 +776,16 @@ def triton_kernel_fused_experts(
             a_ragged,
             None,
             None,
-            None,
+            routing_data.scatter_indx,
             quant_config.w2_precision,
             gammas=None if apply_router_weight_on_input else gammas,
         )
+        out = output_tensor.view(M, K)
+        if _TK38_SCATTER_COMBINE:
+            masked_moe_sum(y.view(M, topk, K), None, out)
+            return out
         acc = torch.zeros((M, K), dtype=torch.float32, device=hidden_states.device)
         acc.index_add_(0, gather_tok.to(torch.int64), y.to(torch.float32))
-        out = output_tensor.view(M, K)
         out.copy_(acc.to(out.dtype))
         return out
```

## 性能声明

- **Throughput**(`vllm bench throughput --model openai/gpt-oss-20b --max_model_len 4096 --dataset-name sharegpt --num_prompts 1000 --tensor_parallel_size 1 --gpu-memory-utilization 0.8`,单卡,每 build 1 cold + 2 warm,取 warm-best):R9700/gfx1201(main c0a41721c7):v0.29 12.22 → main 10.57 → 本 PR **12.43 req/s**(+17.6% over main,超 v0.29 1.7%);W7900/gfx1100(main db3b0f0fae):v0.29 7.86 → main 7.69 → 本 PR **7.84**(+2.0%,与 v0.29 持平;gfx1100 上 combine 本身便宜,每 call 约 45 µs vs 约 590 µs)。
- **rocprofv3 kernel time**(R9700,main c0a41721c7,同 benchmark 一轮 warm run,s):combine `indexFuncLargeIndex<float, …, ReduceAdd>` 20.08 → `_masked_topk_sum_kernel` 0.47;fp32 fill/copy/cast 2.51 → 1.03;MoE GEMM1+GEMM2 60.02 → 62.53;总 GPU kernel 时间 103.12 → 84.92。v0.29 对照:GEMMs 63.93 s、`matmul_ogs` `_reduce` 0.61 s、总 87.36 s。
- **Isolated MoE layer**(真实 gpt-oss-20b layer-0 权重,R9700,中位 µs):combine 对 M=1…8192 全部与 `index_add_` 路径 bitwise 相等;combine M=1: 25→8、M=64: 133→9、M=256: 466→12、M=1024: 1841→34、M=8192: 15206→425;整个 MoE layer M=8192: 53998 → 40030 µs。
- **GSM8K**(`lm_eval --model vllm --tasks gsm8k --num_fewshot 5`,1,319 题,R9700,各跑两次——同版本重跑在 91–93% 的 generations 上不同):main flexible/strict 0.3685/0.2570、0.3677/0.2654;本 PR 0.3571/0.2540、0.3685/0.2555——超出 run-to-run noise 的变化为零。

## 验证与回归风险

- `pytest tests/kernels/moe/test_gpt_oss_triton_kernels.py -k "not test_equiv"` 在 gfx1201 通过;gfx1100 上 `test_make_routing_data_tk38_matches_reference` 6/6 通过。
- `test_equiv` 在 gfx1201 上带不带本 PR 都失败(ROCm 报 capability 12.0,套用 NVIDIA SM12x `is_persistent` 约束)。
- 本地等价性测试(scatter combine vs `index_add_`,含/不含带 −1 slots 的 expert map)在 gfx1201 与 gfx1100 各 6/6 通过;未随 PR 提交,因为 CI 无 RDNA runner。
- `VLLM_GPU_SYNC_CHECK=error` 下运行 patched 路径无同步错误。
- 风险:仅 RDNA 经过验证,CDNA/NVIDIA 的解除门控待验证;`OAITritonExperts` 仍走旧 combine;新 Triton specialization 的首次编译日志。

## 关联

- #55934(OAI Triton MoE 移植到 triton_kernels 3.8,引入 `index_add_` combine);#64331(同作者的 RDNA MXFP4 matmul config pin bump,同为 gpt-oss/RDNA 系列)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
