# PR #42076: [Bugfix] Fix GDN KKT precision loss on Hopper GPUs by aligning tl.dot operand layout with WGMMA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42076](https://github.com/vllm-project/vllm/pull/42076) |
| 作者 | Kermit-C |
| 合并日期 | 2026-05-09 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Hopper 上 GDN KKT kernel 的 tl.dot 因多余 .to() 扰乱 WGMMA 操作数布局产生 NaN,gsm8k 归零;调整 .to() 到转置操作数修复。

## 问题现象
Hopper 架构 GPU(如 H20,sm_90)上,GDN 模型的 `chunk_scaled_dot_kkt_fwd_kernel` 出现精度丢失,产生垃圾输出——lm_eval gsm8k exact_match 为 0(Qwen3.5-9B FP8、`gdn_prefill_backend=triton`,5-shot:修复前 flexible-extract 0±0、strict-match 0±0;修复后 0.8560±0.0097 / 0.8704±0.0093)。确定性诊断(diag_gdn_h20.py)显示修复前 H20 上所有 H=16 配置的 KKT/TRIL/w 子核输出 max_diff=NaN,H=4 正常;Ada(4060ti, sm_89)及更早架构不受影响,故开发期未暴露。

## 根因
正文披露完整根因链:#37813 优化 GDN conv 路径时引入副作用——传给 `chunk_scaled_dot_kkt_fwd_kernel` 的 `g` 张量由 float32 变为 bf16;原有 `b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))` 中 `.to(b_k.dtype)` 在数据类型上是恒等转换(b_kb、b_k 均为 bf16),但在 Triton IR 中会插入 convert_layout 改变操作数 A 的寄存器布局;Hopper 上 Triton 将 tl.dot 下沉为 WGMMA 指令,对操作数布局高度敏感,布局被扰动后硬件额外重排导致点积累加器精度丢失,误差经 `exp(b_g_diff)` 传播最终产生 NaN;float32 时代 `exp(b_g_diff)` 的高精度乘法会掩盖该误差,bf16 后被放大超过 NaN 阈值。Ada 及更早架构用 HMMA/WMMA,编译器会消除冗余布局转换、两写法生成相同 PTX。

## 修复方案
把 `.to()` 从非转置操作数(A)移到转置操作数(B),让 Triton 将 transpose + 类型转换合并为单一布局变换,直接产出 WGMMA 原生期望的寄存器布局(该写法在其他 FLA kernel 如 chunk_delta_h.py 中已一致使用):
```python
# Before (buggy):
b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
# After (fixed):
b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))
```
仅改 `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py` 第 86 行一处。验证:diag_gdn_h20.py 各子核两次运行输出完全一致(max_diff=0.000000e+00),gsm8k 恢复 0.85+。

## 影响与回归风险
- 影响窗口:自 #37813 将 `g` 改为 bf16 起(2026-05-09 修复),Hopper(sm_90)GPU 上 `gdn_prefill_backend=triton` 的 GDN 模型(Qwen3.5 等线性注意力家族)
- 回归面:仅该一处 tl.dot 写法;非转置操作数不再做恒等布局转换,所有架构的 PTX 生成路径可能变化(Ada 上两写法已被证明等价);其他使用同类 `x.to()` 模式的 FLA kernel 不在本次修改范围。

## 关联
- [#37813](https://github.com/vllm-project/vllm/pull/37813)(引入 g→bf16 副作用的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
