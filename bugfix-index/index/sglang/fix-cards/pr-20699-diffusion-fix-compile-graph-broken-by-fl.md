# PR #20699: [Diffusion] Fix compile graph broken by flashinfer rope

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20699](https://github.com/sgl-project/sglang/pull/20699) |
| 作者 | BBuf |
| 合并日期 | 2026-03-16 |
| labels | run-ci, diffusion |
| 热度 | 👍 1 · 💬 4 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | regression |

## 一句话结论
flashinfer rope 直接函数调用使 torch.compile 在 rotary utils 处 graph break;把 flashinfer rope 注册为 compile-friendly custom op,graph break 2→0,Qwen Image denoise 5ms763us→4ms592us。

## 问题现象
diffusion 模型开启 `--enable-torch-compile` 时,`rotary_embedding/utils.py` 触发 torch.compile graph break(正文实测每个受测模型 2 处),rope kernel 不被 torch compile graph 覆盖(main 的 profile 图注明 "rope kernel is not covered by torch compile graph")。影响:编译图断裂、denoise 变慢;Qwen-Image-2512 上 denoise latency main 为 5ms763us。受测模型:qwen-image-2512、qwen-image-edit-2511、FLUX.1-dev、FLUX.2-dev、Z-Image-Turbo(同一组 `sglang generate` 命令,50 步、1024x1024、seed=42)。

## 根因
正文披露(以 "wraps `apply_flashinfer_rope_qk_inplace` as a compile-friendly custom op" 的对比表述):`apply_flashinfer_rope_qk_inplace` 在函数体内直接 import 并调用 flashinfer 的 `apply_rope_with_cos_sin_cache_inplace`,这不是 torch.compile 可追踪的 custom op,编译器在此处断图,形成来自 `rotary_embedding/utils.py` 的 graph break。

## 修复方案
改 `python/sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py`:

1. 模块加载时(CUDA 平台、try/except 保护)把 flashinfer 的 `apply_rope_with_cos_sin_cache_inplace` 通过 `register_custom_op_from_extern` 注册为 custom op:`op_name="flashinfer_apply_rope_with_cos_sin_cache_inplace"`、`mutates_args=["query", "key"]`;
2. `apply_flashinfer_rope_qk_inplace` 不再在函数内 import,改为调用已注册的 `flashinfer_apply_rope_inplace`:

```diff
-    try:
-        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
-    except ImportError:
+    if flashinfer_apply_rope_inplace is None:
         # Triton fallback for AMD/ROCm where FlashInfer is not available
...
-    apply_rope_with_cos_sin_cache_inplace(
+    flashinfer_apply_rope_inplace(
```

3. flashinfer 不可用时仍走原 Triton fallback(AMD/ROCm)。

验证:同一组 5 个命令在 main 与 patch 版分别跑 perf(读 perf dump 的 `DenoisingStage` 计时)与 graph-break 诊断(`TORCH_LOGS=graph_breaks` 计行)。结果:各模型来自 `rotary_embedding/utils.py` 的 graph break 2→0,总 graph-break 数全降,denoise latency 持平或改善,最大收益在 Qwen Image(5ms763us→4ms592us,profile 确认 rope kernel 已被编译图覆盖)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:multimodal_gen 的 rope 路径。CUDA + flashinfer 可用时 rope 改经 custom op 分发(带 `mutates_args` 声明,in-place 语义需被 compile 正确处理);flashinfer 缺失/非 CUDA 时行为不变(Triton fallback);flashinfer import 失败的检测从"每次调用时"前移到"模块导入时"。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
