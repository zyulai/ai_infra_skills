# PR #45000: [Perf][ROCm] Fix GDN KKT warmup regression on RDNA by avoiding fp32 tl.dot

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45000](https://github.com/vllm-project/vllm/pull/45000) |
| 作者 | nemanjaudovic |
| 合并日期 | 2026-07-14 |
| labels | bug, ready, rocm |
| 热度 | 👍 2 · 💬 7 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
PR #42076 后 RDNA 上 GDN triton kernel 的 tl.dot 被降为软件 fp32 matmul,warmup 从 ~150s 涨到 ~2400s;RDNA 上改用 k 原生 dtype 走 WMMA。

## 问题现象
在 RDNA GPU 上运行使用 GDN triton kernel 的 Qwen3.5-9B 模型时,warmup 时间从约 150s 恶化到约 2400s(约 16 倍)。作者对各级 IR 与编译时长做了详细测试,确认回归发生在 amdgcn 阶段,该阶段差异约 190x。

## 根因
正文披露:PR #42076 之后,`tl.dot()` 的操作数被 cast 成 fp32,导致该运算被 lower 成软件实现的 fp32 matmul,而不是 WMMA 指令。diff 注释进一步说明机制(与正文一致):RDNA(gfx11xx/gfx12xx)的 WMMA 只接受 16-bit/int 输入,加宽(如 fp32)的 tl.dot 会被降级为软件 matmul,造成约 190x 的 amdgcn 阶段膨胀。

## 修复方案
改 `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py` 的 `chunk_scaled_dot_kkt_fwd_kernel`:
- 新增 `CAST_DOT_TO_K_DTYPE` constexpr 编译期开关,模块级按平台初始化(仅 ROCm 且 `on_gfx1x()` 即 RDNA 时为 True)。
- RDNA 上把两个操作数 cast 回 k 的原生存储 dtype(bf16/fp16),使 tl.dot 走快速 WMMA;其他 GPU 保持原逻辑(保留 beta 缩放操作数的提升精度,走 WGMMA/MFMA),即 RDNA 之外的路径完全不动。

```diff
-        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))
+        if CAST_DOT_TO_K_DTYPE:
+            # RDNA: force operands to k's native dtype so WMMA is used.
+            b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
+        else:
+            # Keep the promoted precision of the beta-scaled operand (WGMMA/MFMA).
+            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))
```

测试:修改后 Qwen3.5-9B warmup 回到约 150s;gsm8k benchmark 验证 RDNA 上精度保持。

## 影响与回归风险
- 影响窗口:PR #42076 合入之后到本修复之前的版本。
- 回归面:仅 `chunk_scaled_dot_kkt_fwd_kernel` 的 RDNA 分支;RDNA 上 tl.dot 从 fp32 提升精度改为 bf16/fp16 原生精度(正文称 gsm8k 精度保持),非 RDNA GPU(含 CDNA、NVIDIA)代码路径零改动。

## 关联
- [PR #42076](https://github.com/vllm-project/vllm/pull/42076)(回归引入者)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
