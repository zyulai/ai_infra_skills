# PR #21019: [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-03-23 |
| 分类 | Attention / Kernel |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 3 个文件(1 新增 2 修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/21019 |

## 一句话结论

用 Triton kernel 融合 Qwen3.5 GDN 投影后的 split/reshape/cat,H200 上输出吞吐 +7.4%、Mean TTFT -10.8%。

## 动机

前序 PR #19321 已为 Qwen3-Next 的 GDN 融合了 qkvz_proj 与 ba_proj,但其 Triton kernel 读的是 interleaved 布局——Qwen3-Next checkpoint 直接存 fused `in_proj_qkvz`。Qwen3.5 的 checkpoint 则把 `in_proj_qkv` 与 `in_proj_z` 分开存,经 `stacked_params_mapping`(q/k/v shard_id 0/1/2,z 为 3)由 `MergedColumnParallelLinear` 按 output_sizes 放入连续区域,matmul 输出为 contiguous 的 `[all_q | all_k | all_v | all_z]`,旧 kernel 无法读取。此前 Qwen3.5 用 4 个独立投影层加 torch split/reshape/cat,launch 次数与中间张量分配都多,需要一个新的 contiguous 版 kernel。

## 优化点

新增 `fused_qkvzba_split_reshape_cat_contiguous` Triton kernel(grid `(batch*seq_len, num_heads_qk)`,num_warps=1、num_stages=3),单个 kernel 完成 split(qkvz → q/k/v/z,ba → b/a)+ reshape + cat,输出格式与 interleaved kernel 完全一致(下游消费者不变)。同时把投影层从 `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a` 四层合并为 `in_proj_qkvz`(output_sizes=[key_dim,key_dim,value_dim,value_dim])与 `in_proj_ba` 两层,prefill 与 decode 都减少 kernel launch 和中间分配。权重加载用 `_make_packed_weight_loader` 包装原 loader:拦截 tuple shard_id(`(0,1,2)` 映射 in_proj_qkv、`3` 映射 in_proj_z、b/a 分别映射 ba 的 shard 0/1),按 checkpoint 侧 split 大小切块后委托给原 int-shard loader,同时兼容 packed 与 split 两种 checkpoint,并覆盖 FP8 的 `weight_scale_inv`/`weight_scale`/`input_scale`。另外 `_forward_input_proj` 在 capture 模式、短序列下用 alt_stream 双流并行两个投影;qwen3_next.py 的 interleaved kernel 移入 `sglang.jit_kernel.triton.gdn_fused_proj` 共享。forward 顺序:ratio ∈ [1,2,4] 且非 CPU 走 fused kernel,CPU AMX 走 sgl_kernel CPU 版,否则回退 torch 路径。

## 关键改动

```text
python/sglang/jit_kernel/triton/gdn_fused_proj.py: 新增 contiguous/interleaved 两个 fused kernel
python/sglang/srt/models/qwen3_5.py: 四投影合一、packed weight loader、fused kernel 接入
python/sglang/srt/models/qwen3_next.py: interleaved kernel 迁出至共享 jit_kernel
```

```python
if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        projected_states_qkvz, projected_states_ba, ...)
```

## 性能声明

H200,Qwen/Qwen3.5-35B-A3B,TP2,bench_serving random(input 256 / output 4096,200 prompts,rate 10),对比 main:

- Output throughput:3240.71 → 3481.08 tok/s(+7.4%);Request throughput 1.60 → 1.71 req/s(+6.9%);Total token throughput 3443.59 → 3699.00 tok/s(+7.4%)
- Mean TTFT:25876.75 → 23071.14 ms(-10.8%);Mean E2E 57949.06 → 52732.48 ms(-9.0%);Mean TPOT 16.16 → 14.95 ms(-7.5%);P95 ITL 20.86 → 14.36 ms(-31.2%)
- 精度(lm_eval gsm8k,5-shot,num_concurrent 109):main flexible-extract 0.8476 / strict 0.8347 → PR 0.8491 / 0.8340,基本持平

## 验证与回归风险

精度验证为 gsm8k 对比与一次 OpenAI chat 接口输出抽查;PR 未添加新单测(Checklist 未勾选)。风险:`_make_packed_weight_loader` 包装了 weight 与 FP8 scale 参数的 loader,若 tuple shard 切块逻辑出错影响 checkpoint 加载正确性;fused kernel 仅覆盖 v/k 头数比为 1/2/4 的情况,其余走 torch fallback;`_forward_input_proj` 的双流仅限 capture 模式且受 `disable_piecewise_cuda_graph` 与 token 阈值控制。

## 关联

前序:PR #19321(Qwen3-Next GDN qkvz/ba 融合,interleaved kernel 的来源);本 PR 将该 kernel 移至 `jit_kernel/triton/gdn_fused_proj.py` 供两个模型共享。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
