# PR #60801: [Kernel][XPU] Mamba2 SSD: tensor-descriptor path for _chunk_scan_fwd_kernel and _chunk_state_fwd_kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @afierka-intel |
| 合并 | 2026-11-16 |
| 分类 | Attention / Kernel(tier: major) |
| Label | intel-gpu, ci/build |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py、ssd_chunk_state.py、vllm/triton_utils/tensor_descriptor.py、vllm/envs.py、tests/kernels/mamba/test_mamba_ssm_ssd.py 等 |
| 链接 | https://github.com/vllm-project/vllm/pull/60801 |

## 一句话结论

给 Mamba2 SSD prefill 的两个热点 Triton kernel(`_chunk_scan_fwd_kernel`、`_chunk_state_fwd_kernel`)加 tensor-descriptor(TD)路径(XPU 默认开,其他平台 `VLLM_TRITON_USE_TD=1` opt-in):Intel 2D block IO 下 `_chunk_scan_fwd_kernel` 单 kernel 最快 2.07x,`vllm bench latency` 4x2048 在 B60/B70 分别 −8.5% / −18.7%。

## 动机

Mamba2 SSD prefill 的大部分 SSD 时间花在这两个 kernel,它们用带 mask 的指针算术做 2D tile 的 load/store。TD 描述符在 Intel XPU 下 lower 成 2D block IO(NVIDIA sm90+ 为 TMA),零填充代替 load mask、输出走 descriptor store;TD 关闭时生成代码不变。此前单 kernel 尝试 #47084(`_bmm_chunk_fwd`)与 #45816(`_chunk_cumsum_fwd`)因端到端收益平淡被关闭。

## 优化点

1. 两个 kernel 的 tile 经 `tl.make_tensor_descriptor`;`USE_TD` 加入 autotune key;无初始状态的序列在 TD 分支跳过 `prev_states` dot(acc 已为零,合并 zeros tile 实测更慢)——这是 scan 收益的主要来源(TD 不跳过时仅 1.18x)。
2. `tensor_descriptor_compatible` 守卫:非单位内层 stride、基址非 16B 对齐、末维宽度 <64B 或非 16B 倍数、外层 pitch < max(64B, 宽度) 或非 16B 倍数时整次 launch 回退指针路径(64B 是 Intel 2D block IO 最小值)。
3. `set_triton_allocator(x.device)` 每次 launch 调用(Triton allocator 是 per-thread ContextVar,DBO 在新线程上跑 forward)。
4. 新测试 37 个(`-k ssd_td`):TD 与指针路径在相同 pinned tile 下对比、守卫逐操作数回退、launch spy 断言实际路径。

## 关键改动

```python
# vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py
+    if USE_TD:
+        # Descriptors zero-pad outside `shape`, which replaces the masks.
+        C_desc = tl.make_tensor_descriptor(
+            C_ptr, shape=[chunk_size_limit, dstate],
+            strides=[stride_C_seqlen, 1],
+            block_shape=[BLOCK_SIZE_M, BLOCK_K_DSTATE],
+        )
+        if HAS_INITSTATES or (seq_idx == seq_idx_prev):
+            for k in range(0, dstate, BLOCK_K_DSTATE):
+                C = C_desc.load([pid_m * BLOCK_SIZE_M, k])
+                prev_states = tl.trans(
+                    prev_states_desc.load([pid_n * BLOCK_SIZE_N, k])
+                ).to(C_ptr.dtype.element_ty)
+                acc += tl.dot(C, prev_states)
+            acc *= scale_m[:, None]
+    elif BLOCK_SIZE_DSTATE <= 128:
```

## 性能声明

数字均照抄 PR 正文(未独立核实)。复现这些结果需要 #60800(SSD autotune warmup 扩到最多 32 chunks);kernel 级数据为"all tiles timed, median of 3 reps"。

- Kernel 级:`_chunk_scan_fwd_kernel` Nano-4B 4x2048 B60:7186 → 3467 µs(**2.07x**);Nano-4B cases B70:**2.26x**;geomean over 20 realistic cases:B60 **2.10x**,B70 **1.78x**。`_chunk_state_fwd_kernel`:1993 → 1092 µs(**1.82x**);B70 **1.57x**;geomean B60 1.25x,B70 1.15x。(20 cases = 4 模型形状 × 5 batch 形状。)
- `vllm bench latency`(Nemotron-3-Nano-4B-BF16, TP=1, seed 42, output-len 1):Arc Pro B60 2048x4:905 ms → 828 ms (−8.5%, 9 samples all faster);8192x1:935 → 862 (−7.8%);128x1:38.2 → 36.2 (−5.3%)。Arc Pro B70 2048x4:−18.7% (−10.4 … −29.6%, 7 samples);8192x1:−15.7%;128x1:−1.5% 与 +4.8%(mixed)。对照"main + 同样 warmup 改动":B70 4x2048 为 −8.3 … −9.4%(7 samples),1x8192 为 −8.4 … −8.9%;128 tokens 时 PR 在 B70 上慢 3–8%。
- `vllm bench serve`(random 2048/128, 200 prompts, concurrency 16,main vs PR with the warmup change):B60 mean TTFT −12.1%,p99 TTFT −11.9%,TPOT −4.9%,output tok/s +6.0%;B70(3 reps):−23.6% / −22.8% / −11.0% / +14.6%(B70 main 吞吐在 server start 间波动 8%:331–359 tok/s,PR 波动 1%,per-rep gain +9.8% 到 +18.7%)。
- gsm8k 5-shot(全量 1319,lm_eval):B60 flexible-extract 0.8476 vs 0.8476、strict-match 0.8453 vs 0.8461;B70 0.8484 vs 0.8476、0.8461 vs 0.8431。

## 验证与回归风险

- `-k ssd_td`(37 tests):B60 37/37;H200 NVL/H200 34 passed、3 skipped(sm90 丢弃 pinned chunk-scan config 的 tile 用例,其 `-sm90` 变体运行);B200 未复跑(此前 34/34)。全文件 128/128(TD on, TD off 各平台均通过)。
- Launch spy 确认 PR 上两个 kernel 全部 `USE_TD=True`、main 上 `False`;TD off 时 H200 NVL 与 B200 上全部 autotune config 的 TTGIR/PTX(69 scan / 28 state 变体)与 main 完全一致。
- 回归风险:CUDA/ROCm 的 opt-in 路径仅由单测验证(CUDA sm90/sm100),无精度或端到端测量,ROCm 未测试;Falcon-H1-0.5B 的 TD state tile 比 main 最优慢 1.04–1.13x,声明以端到端数字为准;与 #58592(CUDA mamba CI step 拆分)重叠,后落者 rebase;依赖 #60800 稳定 autotune 选择。

## 关联

RFC #42545(TD 工作系列,共享 `VLLM_TRITON_USE_TD`);依赖 #60800;被关闭的先例 #47084、#45816;重叠 #58592;#61718(sm90 drops config)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
