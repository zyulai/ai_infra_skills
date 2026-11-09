# PR #54849: [ROCm][CI][The Rock 10] Fix (MI355) Quantized Models failure on The Rock 10 with Triton 3.8.x

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54849](https://github.com/vllm-project/vllm/pull/54849) |
| 作者 | rasmith |
| 合并日期 | 2026-09-17 |
| labels | rocm, ready, quantization |
| 热度 | 👍 0 · 💬 37 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
MI355 上 Triton 3.8 默认启用 async copy,num_stages=3 编译失败致引擎起不来;gfx950+3.8.x 时 BLOCK_K=256 配置降为 num_stages=2。

## 问题现象
The Rock 10(Python 3.12、Ubuntu 22.04、Triton 3.8.x)上,`(MI355) Quantized Models` 测试组的 mxfp8 用例全部失败,报 `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`,涉及 `test_mxfp8_logprobs[dense]/[moe]` 与 `test_mxfp8_generation[dense]/[moe]` 四个用例——引擎核心初始化直接失败。验证手段:同一测试在 MI355 上设 `TRITON_HIP_USE_ASYNC_COPY=0` 即通过。

## 根因
正文披露:在 MI355 上 `TRITON_HIP_USE_ASYNC_COPY` 默认启用,而其他 MI 平台均默认禁用;正是这一默认差异使 mxfp8 kernel 在 Triton 3.8.x 下编译/初始化失败。至于失败的具体资源机制,diff 注释给出解释(由 diff 推断):async copy 的额外 LDS 缓冲使 BLOCK_K=256 时容纳不下 num_stages=3。该默认值只在 3.8.0 的 gfx950 上生效,故问题范围被限定在 MI350 系(gfx950)+ Triton 3.8.x 组合,3.7.1 不受影响。

## 修复方案
改 `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py`:模块级定义 `_BK256_STAGES = 3`,仅当 `triton.__version__` 为 3.8.x 且 ROCm 平台、`on_gfx950()` 为真时置为 2;`_select_cfg` 中两处返回 BLOCK_K=256 的分支(occ>=256 的大 M 大 K 分支与 qkv-class/deep-K 分支)由硬编码 `3` 改为引用 `_BK256_STAGES`:

```diff
+_BK256_STAGES = 3
+if triton.__version__.startswith("3.8") and current_platform.is_rocm():
+    from vllm.platforms.rocm import on_gfx950
+
+    if on_gfx950():
+        _BK256_STAGES = 2
```

作用域刻意收窄:只有 gfx950 + 3.8.x 用 2,其他组合维持 3,正文确认 3.7.1 无回退。附带收益:Triton 3.8.x 下 num_stages=2 反而更快,`_mxfp8_linear_kernel` 小测显示四个 shape 的 µs 下降 11.1%-18.3%(如 8192,4096,1024 从 74.3 到 62.8)。作者在 MI300 与 MI350 上分别用 3.7.1 与 3.8.0 验证了各组合取到正确的 num_stages;`pytest -sv models/quantization/test_mxfp8.py::test_mxfp8_logprobs` 2 passed。diff 中留有 TODO:待 #50605 合并后移除 3.8 判断。

## 影响与回归风险
- 影响窗口:Triton 3.8.x + gfx950(MI350/MI355)组合;3.7.1 与非 gfx950 平台不受影响。
- 回归面:仅 mxfp8 ROCm 原生线性 kernel 的 tile 配置选择,数学不变、只改流水级数;改动按平台/版本条件生效,MI300(非 gfx950)与 3.7.1 路径保持原值。后续升级到 #50605 的方案时需按 TODO 清理该条件分支。

## 关联
- [#50605](https://github.com/vllm-project/vllm/pull/50605)(TODO 中提到的后续修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
