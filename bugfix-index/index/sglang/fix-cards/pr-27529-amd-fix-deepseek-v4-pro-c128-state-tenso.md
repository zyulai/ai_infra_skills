# PR #27529: [AMD] Fix DeepSeek V4 Pro c128 state tensor dtype mismatch error and c4_sparse_raw_indices attribute error in cuda graph phase

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27529](https://github.com/sgl-project/sglang/pull/27529) |
| 作者 | At1a8 |
| 合并日期 | 2026-06-10 |
| labels | deepseek, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 14 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
DeepSeek V4 Pro 在 AMD 上运行时 compress JIT kernel 因张量 dtype 不一致直接崩溃、c4_sparse_raw_indices 属性缺失报错;kernel 按 per-tensor dtype 模板化并补齐属性。

## 问题现象
按正文给出的 server 命令(TP8/DP8、`--attention-backend dsv4`、`--page-size 256`、`SGLANG_USE_AITER=1` 等一组优化开关)运行 DeepSeek-V4-Pro 时出现三类报错(正文附截图):1) compress JIT kernel(c128_v2.cuh)要求 `kv_score_buffer`/`kv_score_input`/`ape` 同 dtype,不一致直接崩溃,报错为 `dtype=float32 not in allowed [bfloat16]`;2) 开启 `SGLANG_OPT_USE_JIT_NORM` 时 jit norm 出现类似错误;3) `c4_sparse_raw_indices` attribute error(报错发生在 cuda graph 阶段,见标题)。

## 根因
正文披露:c128 compress JIT kernel 要求 `kv_score_buffer`/`kv_score_input`/`ape` 三个张量 dtype 相同,而运行时它们的 dtype 并不一致(如 fp32 状态池对 bf16 输入),喂给 kernel 前没有统一,因此直接崩溃。对 `c4_sparse_raw_indices` attribute error,正文只给出报错与修复位置(deepseek_v4_backend_hip_radix.py 中补充该属性),未披露更详细的根因;diff 摘录中也未包含该文件,无法进一步描述机制。

## 修复方案
按正文 Modifications 与 diff:1) `c128_v2.cuh`/`c4_v2.cuh` 的 compress kernel 按 per-tensor dtype 模板化(`BufFloat`/`InFloat`/`ApeFloat`/`OutFloat`),每个输入在 load 时 cast 到 fp32,不做 host 侧 `.to()`;`c128_forward`/`c4_forward` 内 kv/score 寄存器改为 fp32 数组,`kv_score_buffer` 用独立的 `BufFloat` 指针类型;2) `fused_norm_rope_v2.cuh` 为 fp32 RMSNorm weight 引入独立 `WeightFloat` 模板参数;3) `compress.py` 构建 JIT 模块时传入 per-tensor dtype:

```python
module = _jit_compress_module(
    head_dim,
    kv_score_buffer.dtype,
    kv_score_input.dtype,
    out.dtype,
    compress_ratio,
)
```

4) `compressor.py` 的 `apply_ape_hotfix` 中,`_use_aiter` 时把 `ape` 与 `norm.weight` 转成 bfloat16。注:Modifications 提到的 `fused_norm_rope_v2.cuh` 与 `deepseek_v4_backend_hip_radix.py` 两处改动未出现在 pack 的 diff 摘录中。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:c128/c4 compress 与 fused norm+rope 的 JIT kernel 模板实例化与签名(需按新 dtype 参数重新编译缓存);`apply_ape_hotfix` 在 aiter 路径改变 ape/norm weight 的 dtype;dsv4 HIP radix attention backend 的 sparse indices 属性。修复后 GSM8K Accuracy 0.945、Invalid 0.000。

## 关联
- https://github.com/sgl-project/sglang/pull/27525

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
