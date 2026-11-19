# PR #50805: [Refactor] kernels: Fix silent failures and prune dead code in the kernels package (3/4)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#50805](https://github.com/sgl-project/sglang/pull/50805) |
| 作者 | BBuf |
| 合并日期 | 2026-11-18 |
| labels | quant, amd, lora, hicache, sgl-kernel, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

kernels 包(3/4)清理 PR(77 文件,+390/-986):修复了一批"永远不生效的检查"导致的静默失败——CPU MoE MXFP4 scale 形状检查形同虚设、int8 对照把张量和自己比较、bench verify 永远打印 mismatch、包导出可能返回 module 而非函数、架构门控写在裸 capability major 上等,同时删除大量零引用死代码。

## 问题现象

正文按领域列出的真实 bug(每项均有"无修复则失败、有修复则通过"的回归测试):

- MiniCPM-SALA `get_block_table`:包通过 module `__getattr__` 惰性导出该函数;一旦同名子模块先被导入(registry 目标和 kernel 测试正是如此),`from sglang.kernels.ops.attention.minicpm_sala import get_block_table` 返回的是子模块,MiniCPM backend 调用时抛 `TypeError: 'module' object is not callable`。
- CPU MoE MXFP4(`fused_experts_cpu`,`aot/csrc/cpu/moe.cpp`):`TORCH_CHECK(w1s.numel(), E * 2 * N * K >> 5)` 把期望大小当作消息传入,尺度大小从未被检查;错误大小的 `w1_scale`/`w2_scale` 会越界读取且不报错。
- CPU MoE INT4 W4A8:bf16/fp16 路由权重抛 `RuntimeError: expected scalar type Float but found BFloat16`;该分支读的是 `topk_weights` 而非其他量化方法都使用的 float32 拷贝。
- `sgl_kernel.test_utils.assert_all_close_or_tiny_diff`:int8 分支计算的是 `a - a`,int8 输出从未与参考值比较过,任何 int8 量化误差都能通过测试。
- `aot/benchmark/bench_rmsnorm.py --verify_only`:`calculate_diff` 没有 return,每次都打印 `mismatch`。
- HPC-Ops bf16xfp32 GEMM 探测:用裸 capability major == 9 判断,而 ROCm gfx950 也报告 (9,5);若 `hpc` 可导入会错误路由到 sm90a kernel(当前潜伏:`hpc` 只发 CUDA wheel)。
- `sampling.softcap_row_argmax` 的 `KernelSpec` 只标 CUDA,但该 op 有自己的 ROCm 分支且唯一调用方在 HIP 上运行,registry 误报不可用(无运行时影响)。
- LTX-2 `ltx2_ada_values9`:未被读取的 `rows: tl.constexpr` 使 Triton 对每个新 `batch * seq` 都编译一个相同 kernel(每个新 token 数一次 JIT 停顿)。

## 根因

均为代码级缺陷,来源与正文一致:惰性导出未处理同名子模块遮蔽;`TORCH_CHECK` 参数顺序写错导致断言退化为真值检查;测试工具写错减数与被减数;benchmark 函数漏写 return;架构门控未使用 `current_arch().is_sm(9)` 语义(应限定 NVIDIA SM9x);`KernelSpec` 能力清单与实际实现不同步;Triton constexpr 参数未参与计算却触发按形状特化编译。

## 修复方案

- 包在导入期直接导入 `get_block_table` 函数(不再依赖 `__getattr__`)。
- MXFP4 两个 scale 改为 `TORCH_CHECK(numel == expected, msg)`;INT4 W4A8 分支改读 float32 拷贝 `topk_weights_`。
- `assert_all_close_or_tiny_diff` int8 分支改为比较 `a` 与 `b`,±127 处的一步差不计入(两侧都截断,H200 实测核与参考的差异恰为 1 且落在组的 absmax 元素上)。
- `bench_rmsnorm.py` 的 `calculate_diff` 补上 `return all_match`。
- HPC-Ops bf16xfp32 GEMM 探测改用 `current_arch().is_sm(9)`;Hopper bf16 GEMV 的 `_is_sm12x()` 从 "capability >= 12.0" 收紧为仅 family 12;`softcap_row_argmax` 能力改为 `{CUDA, HIP}`。
- 删除 LTX-2 kernel 的未读 `rows` 参数(四个特化的 PTX 前后一致)。
- 死代码清理:零引用 Python 符号(如 `cute_utils` 的 PTX 包装、`dsv4/rope.py` 的 `_yarn_get_mscale` 等)、不可达分支(如 `fused_gather_attn_decode_dsv4_dual_scope` 的 split-K 半边约 160 行、lean decode 未读 constexpr)、JIT C++ 零引用代码(`ngram_corpus` 的 `Param::parse`/`Result::truncate` 等、`vec.cuh` 无法编译的非 fp32 分支)、重复代码合并(`hicache_tma.cuh`/`topk_impl.cuh` 的私有 PTX 包装并入共享 `sgl_kernel/ptx`)、JIT 构建清理(`gelu_tanh_cat` 的硬编码 `-lineinfo`)以及约 12 个 benchmark 脚本与多处陈旧注释。

关键 diff 摘录:

```diff
-      TORCH_CHECK(w1s.numel(), E * 2 * N * K >> 5);
-      TORCH_CHECK(w2s.numel(), E * K * N >> 5);
+      TORCH_CHECK(w1s.numel() == (E * 2 * N * K >> 5), "mxfp4 w1_scale must hold one scale per 32 weights.");
+      TORCH_CHECK(w2s.numel() == (E * K * N >> 5), "mxfp4 w2_scale must hold one scale per 32 weights.");
```

```diff
-          topk_weights.data_ptr<float>(),
+          topk_weights_.data_ptr<float>(),
```

```diff
     elif a.dtype == torch.int8:
-        diff = (a.to(torch.int16) - a.to(torch.int16)).abs()
+        a_i16 = a.to(torch.int16)
+        b_i16 = b.to(torch.int16)
+        diff = (a_i16 - b_i16).abs()
```

## 影响与回归风险

- 无 CLI flag、环境变量或默认值变化;`sgl_kernel` wheel 公共 API 不变(不删导出的 op/wrapper/`__init__` 名)。
- 有意的行为变化:错误大小的 MXFP4 scale 现在报 `RuntimeError`(仓库内调用方总是传对);INT4 W4A8 的 bf16/fp16 `topk_weights` 自动转 float32;`bench_rmsnorm --verify_only` 从永远 `mismatch` 变为一致时打印 `sanity pass`;`torch.ops.sglang.fused_moe_lora` schema 移除 `fully_sharded`(传则报错,无调用方传过);`mhc_fused_post_pre` 对多余参数报 `TypeError`。
- 清理证明方式:全仓库词搜索、H200 上输出 bit-identical/sha256/PTX/SASS 对比、冷构建等;约 200 行新增为测试。
- 两个 CPU MoE 测试在 x86 (AMX) 手工构建的扩展上验证:基线 3 失败/67 通过,本分支 70 全过。
- 正文"For the owners"列出若干已准备但未包含的数值类修复(xAI temperature 在确定性 prefill kernel 与其他 kernel 间不一致、`fuse_scale_shift_kernel` 非阻塞拷贝先读、`infllm_v2` stage 1 用 bf16/128 kernel 服务 fp16 与任意 head dim、`segment_packbits` 未注册、split-KV reduce 重复),需要模型 owner 在真实权重/目标硬件上验证后另行处理。

## 关联

- 系列拆分:#50803(1/4 srt runtime core)、#50804(2/4 srt models and layers)、#50806(4/4 diffusion);四者文件不相交、可任意顺序合并。
- 前序同类清理:#44690、#48366、#48399、#49189;使用 code-cleanup skill(#49163)。
- 相关 RFC:#29630(`kernels.ops` / `kernels.jit` 命名空间)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
