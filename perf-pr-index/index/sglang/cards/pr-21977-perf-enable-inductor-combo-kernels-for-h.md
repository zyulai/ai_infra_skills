# PR #21977: perf: enable inductor combo_kernels for horizontal fusion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jasperjiaguo |
| 合并 | 2026-04-10 |
| 分类 | CUDA Graph / Compiler(classify.py 给出,tier: major) |
| Label | run-ci, piecewise-cuda-graph |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 仅 1 个文件,约 +13 行 |
| 链接 | https://github.com/sgl-project/sglang/pull/21977 |

## 一句话结论

在 piecewise CUDA graph 的 inductor 编译配置中开启 combo_kernels 横向融合,Qwen3-0.6B 每 forward GPU kernel 数从 413 降到 357(-14%)。

## 动机

piecewise CUDA graph + inductor 编译下,形状不同的兄弟算子各自生成独立 triton kernel,QK normalization 这类成对算子(q_norm + k_norm)会产生多份小 kernel,launch 开销在高并发或小模型场景占比放大。PyTorch 2.9 的 inductor 提供 `combo_kernels` 横向融合能力但默认关闭,本 PR 在 SGLang 的编译配置层将其打开。要求 torch >= 2.9.0。

## 优化点

在 `CompilationConfig.__init__` 中新增 `configure_inductor()`:仅当 compiler 为 inductor 时,把 `torch._inductor.config.combo_kernels` 与 `benchmark_combo_kernel` 置 True(combo_kernels 属性不存在时跳过,兼容旧 torch)。combo_kernels 允许 inductor 将形状不同的 sibling 算子横向融合为单个 triton kernel;benchmark_combo_kernel 让 inductor 实测融合前后性能再决定是否采用。效果上 q_norm + k_norm 的 reduction + pointwise kernel 被融合为单 kernel,`split_with_sizes`/`clone` 等中间算子也从 kernel 名单中消失。

## 关键改动

```text
python/sglang/srt/compilation/compilation_config.py: 新增 configure_inductor(),inductor 下开启 combo_kernels
```

```python
def configure_inductor(self):
    """Apply inductor-specific optimizations when using inductor compiler."""
    if self.compiler != "inductor":
        return

    import torch._inductor.config as inductor_config

    # Horizontal fusion for sibling ops with different shapes,
    # e.g. fusing q_norm + k_norm into a single triton kernel.
    if hasattr(inductor_config, "combo_kernels"):
        inductor_config.combo_kernels = True
        inductor_config.benchmark_combo_kernel = True
```

## 性能声明

Qwen3-0.6B FP8 embeddings,H200,PCG(piecewise CUDA graph)inductor,7k tokens:GPU kernels per forward 413 → 357(-14%);每层 QK norm kernel 数 4 → 2;kernel 名中的 `split_with_sizes`/`clone` 消失。吞吐在 60 RPS 下为中性(该负载下 kernel launch 开销不是瓶颈);作者指出 kernel 数下降应有利于更高并发或 launch 开销占比更大的小模型场景。PR 未提供更高负载的实测吞吐数字。

## 验证与回归风险

PR 正文只附 profile 截图佐证 kernel 数下降,未列新增单元测试,也未给精度对比(combo_kernels 属编译期融合,理论上不改数值,但 PR 未注明)。风险:全局开启 combo_kernels 影响 PCG inductor 路径上所有模型的 kernel 编译产物,`benchmark_combo_kernel` 会增加编译期时长;依赖 torch >= 2.9.0,旧 torch 由 `hasattr` 守卫静默跳过;非 inductor compiler 不受影响。

## 关联

同区域:piecewise CUDA graph(PCG)与 torch.compile 相关 PR。PR 正文未列出具体关联编号,关联标记为:无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
