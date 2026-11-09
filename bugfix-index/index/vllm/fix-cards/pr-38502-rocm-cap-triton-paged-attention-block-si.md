# PR #38502: [ROCm] Cap Triton paged attention block size to fix ROCm shared memory OOM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38502](https://github.com/vllm-project/vllm/pull/38502) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-05-10 |
| labels | rocm, ready, ci/build, v1 |
| 热度 | 👍 1 · 💬 9 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论

混合 Mamba 模型把 block_size 撑到 2048,Triton fallback 共享内存超限,现封顶 128。

## 问题现象

混合 Mamba 模型(如 Jamba)为了让 attention 与 Mamba 页大小对齐,把 `block_size` 撑大到 2048。ROCm 自定义 paged attention 内核只支持 16/32、会拒绝该值,于是走到 Triton fallback `kernel_paged_attention_2d`;该内核直接拿 2048 当 tile size,申请 262144 字节共享内存,超出 MI325X 硬件上限 65536 字节,导致 OOM。影响面为 ROCm 上使用混合 Mamba 架构(大 block_size)的模型。

## 根因

按正文披露:混合 Mamba 模型把 block_size 抬到 2048;Triton fallback 内核 `kernel_paged_attention_2d` 把物理 block_size 直接用作 tile size(TRITON_BLOCK_SIZE),据此申请共享内存,2048 对应 262144 字节,超过 MI325X 的 65536 字节上限。正文同时指出修复可行性的依据:该内核已通过 `l_block_idx`/`internal_offsets` 寻址把 tile size 与物理 block size 解耦,因此 tile 封顶不影响正确性。

## 修复方案

- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`:在 `chunked_prefill_paged_decode` 中新增 `MAX_TRITON_BLOCK_SIZE = 128`,`TRITON_BLOCK_SIZE = min(block_size, MAX_TRITON_BLOCK_SIZE) if is_pow2 else 32`,即 tile 封顶 128,避免超共享内存限制。
- 同文件新增 `has_native_kv_cache_layout(key_cache, value_cache)`:用 `stride(0) == shape[1:].numel()` 判断 KV cache 块是否为紧凑(packed)布局;decode 侧对非 2 次幂 block 或非原生布局都强制 `use_custom = False` 走 Triton 路径。
- `vllm/v1/attention/backends/rocm_attn.py`:`do_kv_cache_update` 中原生 HIP C++ 写入路径现在要求 `block_size in (16, 32)` 且 `has_native_layout`;其余情况走 stride-aware 的 `triton_reshape_and_cache_flash`。(由 diff 推断)注释说明原生 reshape_and_cache 内核假设块存储连续,在 hybrid 布局上会写错块,故 cache 更新与 decode 需要一致的 Triton 路径。
- `.buildkite/test-amd.yaml`:CI 侧改为安装 ROCm 版 mamba(`rocm-7.0-v2.3.0`)与 causal-conv1d `v1.6.0`,并把 hybrid 模型测试拆分到 `-m hybrid_model` 等分组运行。

关键 diff 摘录:

```diff
-        TRITON_BLOCK_SIZE = block_size if is_pow2 else 32
+        MAX_TRITON_BLOCK_SIZE = 128
+        TRITON_BLOCK_SIZE = min(block_size, MAX_TRITON_BLOCK_SIZE) if is_pow2 else 32
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:ROCm 注意力路径三处:大 block_size(如 2048)下 Triton decode tile 变小,吞吐可能变化但不再 OOM;hybrid(attention/Mamba)布局的 cache 写入与 decode 被统一强制到 Triton 路径,原生 HIP 路径覆盖面缩小,若 stride 判断有误可能影响常规模型的内核选择;CI 脚本对 hybrid 测试的分组调整可能改变各 shard 的负载分布。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
