# PR #54261: [Kimi-K3][Perf] Make native CUDA AttnRes the SM100 default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-08-31 |
| 分类 | Model-specific Fast Path |
| Label | performance, ready, nvidia, kimi, k3 |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 CUDA kernel(`attn_res_kernel.cu`)、绑定、`_custom_ops.py`、attn_res 分发、测试与新增 benchmark |
| 链接 | https://github.com/vllm-project/vllm/pull/54261 |

## 一句话结论

把 Kimi-K3 的 AttnRes(residual 加 + RMSNorm)在 SM100 上从 Triton 切到原生 CUDA kernel(覆盖全部调用变体),单 token 4 blocks 快 1.49x,大 shape 最高 1.72x。

## 动机

Kimi-K3(hidden_size=7168)的 AttnRes 此前只有"fused-add + output-norm"这一条常见路径走原生 SM100 CUDA kernel,block 边界、无 output-norm、block 写回等变体都退回 Triton;原生 kernel 也只支持 1-8 blocks 且要求 delta/output_norm 必传。不同请求的调用形态(块数、是否写回 block bank、是否归一化)差异使大量调用点落在慢的 Triton 路径,小 batch decode 尤其吃亏。

## 优化点

(1) 把 register/TMEM kernel 扩展到 0-8 blocks、可选 prefix 加法(delta 变 optional)、可选 output 归一化与 block-bank 写回(`block_write_idx`),签名改为 `Tensor? delta`、`Tensor? output_norm_weight`;(2) 所有支持的 block 数都走既有 source-chunk 调度,同时当 delta 与 output_norm 同时存在且无写回时启用编译期折叠的 STATIC_COMMON_PATH,关闭 output-norm 时归一化作为编译期特化直接省掉其 SMEM buffer、barrier 与算术;(3) PDL 优化:`cudaGridDependencySynchronize()` 从 kernel 入口移到首个生产者依赖的 prefix/delta TMA 之前(warp 0 内 `pdl_waited` 标志保证只等一次),并在生产者 TMA 全部发出后 `cudaTriggerProgrammaticLaunchCompletion()`(原来在结尾);(4) Python 分发条件放宽为 dense 行布局 + 0<=num_blocks<=8,SM100 family + 编译了该 op 即走原生,不支持设备与非 dense 行保留 Triton fallback;(5) 新增可复现 CUDA vs Triton 内核 benchmark。

## 关键改动

```text
csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu: kernel 模板重构(STATIC_COMMON_PATH/HAS_OUTPUT_NORM)+ PDL 前移
csrc/libtorch_stable/{ops.h,torch_bindings.cpp}: 绑定签名扩展(delta/output_norm 可选 + block_write_idx)
vllm/_custom_ops.py + vllm/models/kimi_k3/nvidia/ops/attn_res.py: 原生路径成为 SM100 dense 默认
benchmarks/kernels/benchmark_kimi_k3_attn_res.py: 新增 CUDA vs Triton 对比基准
tests/models/kimi_k3/test_attn_res.py: 变体覆盖 + "不回退 Triton" 测试
```

```cpp
+            if (!has_delta && ns + n == N - 1 && !pdl_waited) {
+              cudaGridDependencySynchronize();
+              pdl_waited = true;
+            }
             cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t),
                           plan.bar_ready[chunk_slot]);
...
+      cudaTriggerProgrammaticLaunchCompletion();
```

## 性能声明

NVIDIA B300 SXM6 AC(单卡 GPU 7),PyTorch 2.13.0 / CUDA 13.0,中位微秒(Triton → CUDA):

- 1 token / 4 blocks:12.29 → 8.22 us(**1.49x**);1 token / 8 blocks:14.34 → 10.24 us(1.40x);0 blocks:8.19 us 持平。
- 4096 tokens / 8 blocks:204.77 → 122.88 us(1.67x);8192 tokens / 8 blocks:395.30 → 229.44 us(**1.72x**)。
- M<=16 且有 block 行:CUDA 快 1.20x-1.49x。
- 无 output-norm 特化:M<=16 持平,大 shape 最高快 4.6%(8-block M=1024:40.86 → 38.98 us)。
- 单 token CUDA graph:内核 device 时间 3.67/4.44/5.48 us(1/4/8 block 行),整图 replay 中位 8.03/10.05/10.05 us。

模型评测:完整 moonshotai/Kimi-K3(MXFP4)on GB300,5 种拓扑 GSM8K 95.53%-97.27%、OCRBench 89.2-89.3,PR 判定无精度回退(串行 99.60% 为 250 子集效应)。

## 验证与回归风险

Kimi-K3 专项测试 34 通过/9 平台跳过;benchmark 内含各 block 数与 PyTorch 参考实现的输出/原地修改校验(atol 8e-2);新增 `test_sm100_variants_do_not_fall_back_to_triton` 用 monkeypatch 让 Triton kernel launch 即抛错,验证 0 blocks、delta 可选、写回等变体不再回退。风险面:原生 op 仅在 SM100 + CUDA>=13 编译,其他平台仍走 Triton;精度容差较宽(8e-2 绝对),归一化/非归一化变体的位级差异 PR 未讨论;PDL 等待点后移依赖"生产者 TMA 发出后再放行 dependents"的时序假设。

## 关联

#53524(AttnRes PDL handoff 移入本 PR,剩余范围为 C=1 MoE/router)、#50760(warmup profiles)。同批无其他 Kimi-K3 卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
