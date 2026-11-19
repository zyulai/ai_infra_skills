# PR #67812: [Bugfix][Kernel] Apply beta after the K loop in chunk_scaled_dot_kkt

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JonSnow1807 |
| 合并 | 2026-11-19 |
| 分类 | Attention / Kernel(tier: major) |
| Label | bug, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67812 |

## 一句话结论

修复 #65385:`chunk_scaled_dot_kkt_fwd_kernel` 改为像上游 FLA 一样在 K 循环之后应用 `beta`,而不是在循环内缩放 `k`——Hopper 上 `BK < K` 且 `num_warps >= 4` 的 autotune 配置从"逐次运行结果不同"变为全部 27 个配置确定且与 fp32 参考差 ≤2.8e-7(main 的确定性配置差 1.7e-3),同时 autotuned kernel 在 H100 上快 4-6%。

## 动机

根因:`k` 既经 shared memory 喂 wgmma、又有寄存器用户(`k * beta`),此时 Triton pipeliner 给它的 buffer 少分一个 slot,下一次迭代的 load 会在 async wgmma 还在读它时覆盖之(triton-lang/triton#12738 在 Triton 侧修复该问题)。把缩放移到循环后,`k` 就没有其他用户,竞争消失。受影响者:Hopper 上 Qwen3-Next 与 Qwen3.5 默认用 FlashInfer GDN prefill,本 kernel 在 `--gdn-prefill-backend triton` 时运行;OLMo Hybrid 直接调 vendored `chunk_gated_delta_rule`。

## 优化点

- 循环内 `b_A += tl.dot(b_kb, tl.trans(b_k)...)` 改为 `b_A = tl.dot(b_k, tl.trans(b_k), b_A)`,`USE_G` 分支后统一 `b_A = b_A * b_beta[:, None]`。
- RDNA:dot 现在总取 `k` 的 dtype,`_CAST_DOT_TO_K_DTYPE` 分支(RDNA WMMA 只收 16-bit/int 输入、fp32 `tl.dot` 会被降为软件 matmul,约 190x amdgcn-stage 膨胀)不再需要,连同 `current_platform` 导入一起删除。
- 相对 #65927(剪掉 `BK < 128` 配置)更彻底:K=256 时那 6 个 `BK=128, num_warps >= 4` 配置仍全部竞争;方案来自 #65385 里 pkusp 的建议。

## 关键改动

```diff
-        b_kb = b_k * b_beta[:, None]
-        if CAST_DOT_TO_K_DTYPE:
-            # RDNA: force operands to k's native dtype so WMMA is used.
-            b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
-        else:
-            # Keep the promoted precision of the beta-scaled operand (WGMMA/MFMA).
-            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))
+        # Beta is applied after the loop, as in upstream FLA (see #65385).
+        b_A = tl.dot(b_k, tl.trans(b_k), b_A)

     if USE_G:
         p_g = g + bos * H + i_h + o_t * H
         b_g = tl.load(p_g, mask=m_t, other=0.0)
         b_g_diff = b_g[:, None] - b_g[None, :]
         b_A = b_A * exp(b_g_diff)
+    b_A = b_A * b_beta[:, None]
```

```diff
-# On RDNA (gfx11xx/gfx12xx) WMMA only
-# accepts 16-bit/int inputs, ...
-_CAST_DOT_TO_K_DTYPE = False
-if current_platform.is_rocm():
-    from vllm.platforms.rocm import on_gfx1x
-
-    _CAST_DOT_TO_K_DTYPE = on_gfx1x()
```

## 性能声明

- 确定性(H100 80GB HBM3、torch 2.14.0+cu132、Triton 3.8.0、main 919d45fc3):main 上 27 个 autotune 配置中 25 个逐次运行结果不同(8/12 @K=128、17/18 @K=256,全部 `BK < K` 且 `num_warps >= 4`),其余 29 个(注:pack 原文如此)距 fp32 参考 1.7e-3;本 PR 后 K=128 与 K=256 所有配置每次运行同结果、全部 ≤2.8e-7。
- Kernel 时间(`triton.testing.do_bench`,T=4096、H=32、Hg=16):autotuned 26.6 → 25.0 us @K=128、34.5 → 33.1 us @K=256(约 4-6%);27 个配置几何均值比 0.94 与 0.95。
- GSM8K(5-shot、greedy、1319 题,Qwen/Qwen3.5-0.8B、`--gdn-prefill-backend triton`、`max_num_seqs=1` 保证确定性;200 题重跑与首轮 200/200 相同):main 31.5% → 本 PR 32.2%(19 与 29 题分别只有一方答对,p = 0.19);batch-invariant 模式不支持 GDN,同 build 的 batched 运行在 106-121 个答案上互不相同。

## 验证与回归风险

- 新测试 `tests/kernels/fla/test_chunk_scaled_dot_kkt.py`:每个配置跑 5 次,运行间 bitwise 对比,并对 fp32 参考以 1e-5 比较;main 上 2 failed(3 秒内),本 PR 3 轮 2 passed。
- 既有测试:`pytest -q tests/kernels/fla tests/kernels/mamba/test_gdn_forward_core_split.py tests/kernels/mamba/test_gdn_fused_mtp.py tests/kernels/mamba/test_gdn_output_projection.py` 在 main 与本 PR 上均 1752 passed / 1 skipped。
- 风险(正文明示):CI 中 `tests/kernels/fla` 跑 B200 与 MI355,那里该竞争不出现——Hopper 的修复只能由本 PR 之外的 H100 实测覆盖;RDNA cast 分支被删除属于行为面变化(由"dot 总取 k 的 dtype"承接)。

## 关联

- Fixes #65385;#65927 的 config 剪枝不足以消除竞争;Triton 上游修复见 triton-lang/triton#12738。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
