# PR #30981: fix(hicache): support staged write-back for asymmetric MHA

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30981](https://github.com/sgl-project/sglang/pull/30981) |
| 作者 | HZY-Wade |
| 合并日期 | 2026-07-22 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 31 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
AsymmetricMHATokenToKVPoolHost(K/V head dim 不对称)缺 staged write-back 路径,一直走非 staged fallback;补上按 K/V 各自 stride 分派的 per-buffer staged 内核路径。

## 问题现象
K 与 V head dim 不同的非对称 MHA host pool(`AsymmetricMHATokenToKVPoolHost`)在 HiCache staged write-back 重构后没有得到 staged 路径:控制器侧已正确为 staged write-back 在 CPU 上保留 host indices,但该 pool 的写回仍持续走非 staged 的 split K/V fallback,staged 优化对该类 pool 实际不可用。正文未描述更具体的用户可见故障(如报错或数据损坏)。验证场景为 MiMo-V2.5-Pro-FP4-DFlash / DFLASH / `page_first + kernel + write_through`:8 卡 L20D 上两个相同 1251-token 请求均 HTTP 200,首个 20.58s、第二个 0.28s 且命中 1152 cached tokens,无 HiCache/staged-kernel/scheduler 异常。

## 根因
正文披露:staged write-back 重构只为对称 K/V 的 host pool 提供了 staged 路径;`AsymmetricMHATokenToKVPoolHost` 的 K、V 是 stride 不同的两个独立 buffer,重构时它被落下,继续使用非 staged 的 split K/V fallback(而非补 per-buffer staged 路径),造成该优化在此类 pool 上缺失。

## 修复方案
改 `python/sglang/srt/mem_cache/pool_host/mha.py`:

1. 新增 `_init_write_back_staging_buffers`:仅 `page_first` 布局、非 NPU/XPU/MPS、(CUDA 或 HIP)且 K/V 两种 element_size 均被 `can_use_write_back_jit_kernel` 支持时启用;按 `min(page_num, _WRITE_BACK_STAGING_PAGE_CHUNK) * page_size` 容量分别分配 staging K buffer(`head_dim`)与 staging V buffer(`v_head_dim`),置 `can_use_write_back_jit=True`;
2. `backup_from_device_all_layer` 的 kernel 分支:可用时对 K、V 各调一次单 buffer staged 内核 `jit_transfer_hicache_all_layer_mla_staged_lf_pf`(各自 native stride 由内核特化保留),否则保留原 `transfer_kv_all_layer_mla_lf_pf` 双调用 fallback:

```diff
+            if self.can_use_write_back_jit:
+                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
+                    ptr_src=device_pool.k_data_ptrs,
+                    src_indices=device_indices,
+                    dst_indices=host_indices,
+                    staging=self.staging_k_buffer,
+                    dst=self.k_buffer,
+                    page_size=self.page_size,
+                )
+                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
+                    ...  # V 同理,用 staging_v_buffer / v_buffer
+                )
+            else:
                 transfer_kv_all_layer_mla_lf_pf(
```

3. 测试:单测新增 staged buffer 初始化、staged 分派(K/V 各自 ptr/staging/dst)与"staged 写回-逐层 load 回环保持非对称 K/V 值"精确回环测试(既有测试置 `can_use_write_back_jit=False`);新增 `test_unified_radix_cache_kl_mimo.py` 回归测试(强制 L2 逐出后比对 host load-back 与 prefill replay 的 logprobs KL,阈值 0.01,注册 nightly-8-gpu-h200);`kl_test_utils.py` 加 `routed_dp_rank` 参数。单测 9 passed。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 `AsymmetricMHATokenToKVPoolHost` 的 kernel 写回路径(page_first)。满足条件时改走 staged jit 内核(K/V 各一次),不满足时走原 fallback,后者逻辑未变;对称 MHA host pool 与其他 io_backend(direct 等)不在本次 diff 内。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
