# PR #29702: Fix wrong RMSNorm fallback to old Flashinfer CUDA kernel when in PCG

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-07-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | sgl-kernel, run-ci, run-ci-extra |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/29702 |

## 一句话结论

修复 PCG 场景下 RMSNorm 误回落到 C++ Flashinfer 注册的旧内核,16k tokens/8k hidden 下算子性能 +20%。

## 动机

在 PCG 场景下,RMSNorm 会错误地解析到经 C++ Flashinfer 注册的旧内核。旧 FlashInfer norm 路径在 torch.compile/fullgraph 下还存在编译问题:Dynamo 会追踪进 FlashInfer 的 JIT 模块加载,调用 `Path.exists()`/`os.stat()` 等不可追踪操作导致整图编译失败(sgl-kernel 里为此留有临时守卫,对应 flashinfer#2734)。作者选择在 SGLang 侧把算子注册为 custom op 来绕开,而不是等 FlashInfer 上游处理;PR 正文同时提出未来应彻底删除 C++ Flashinfer 代码。

## 优化点

两处机制改动。其一,`python/sglang/srt/layers/layernorm.py` 不再直接 `from sgl_kernel import rmsnorm` 等四个函数,而是经 `register_custom_op_from_extern` 包装成 SGLang 侧 custom op(op 名 sgl_rmsnorm、sgl_fused_add_rmsnorm、sgl_gemma_rmsnorm、sgl_gemma_fused_add_rmsnorm),in-place 变体显式声明 `mutates_args=["input", "residual"]`,使 PCG/torch.compile 能正确追踪与分发,不再落到旧 Flashinfer 内核。其二,sgl-kernel 的 `elementwise.py` 删除了 `not torch.compiler.is_dynamo_compiling()` 守卫——此前 Dynamo 编译期间会强制回落到内部实现,现在重新允许 FlashInfer norm 路径参与编译。

## 关键改动

```text
python/sglang/srt/layers/layernorm.py: 将 sgl_kernel 四个 norm 函数包装注册为 custom op
sgl-kernel/python/sgl_kernel/elementwise.py: 移除 dynamo 编译期强制回落内部实现的临时守卫
```

```python
rmsnorm = register_custom_op_from_extern(
    _sgl_rmsnorm, op_name="sgl_rmsnorm", out_shape="input"
)
fused_add_rmsnorm = register_custom_op_from_extern(
    _sgl_fused_add_rmsnorm,
    op_name="sgl_fused_add_rmsnorm",
    mutates_args=["input", "residual"],
)
```

## 性能声明

PR 正文仅给出算子级数字:num tokens = 16k、hidden = 8k 时,算子性能提升 20%(附截图佐证)。模型、硬件、精度与 workload 条件 PR 未注明。

## 验证与回归风险

diff 未新增测试;pack 中记录的 CI 状态显示 PR Test (Base) 与 (Extra) 均为失败标记。回归面:layernorm 是几乎所有模型的公共路径,custom op 注册改变 torch.compile/PCG 下的算子分发;删除 dynamo 守卫隐含依赖 FlashInfer 上游修复(flashinfer#2733/#2734,见原守卫注释),若环境内 FlashInfer 版本未含修复,编译期问题可能复现。

## 关联

无。上游依赖:flashinfer-ai/flashinfer#2733 / #2734(见 diff 注释)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
