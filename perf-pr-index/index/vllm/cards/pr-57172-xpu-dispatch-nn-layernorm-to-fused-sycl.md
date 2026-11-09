# PR #57172: [XPU] Dispatch nn.LayerNorm to fused SYCL kernel via CustomOp

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mganczarenko |
| 合并 | 2026-09-30 |
| 分类 | Attention / Kernel (tier: major) |
| Label | intel-gpu, ready, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 15 个文件(layernorm、_xpu_ops 与多个模型接入) |
| 链接 | https://github.com/vllm-project/vllm/pull/57172 |

## 一句话结论

把 bare nn.LayerNorm 经 CustomOp 分派到 XPU 融合 SYCL kernel,GPT-2 等 E2E 吞吐最高 +9.2%。

## 动机

PR 正文 Purpose:vllm-xpu-kernels 已为标准(非 RMS)LayerNorm 提供融合 SYCL kernel(vllm-xpu-kernels#577:`layer_norm`、`fused_add_layer_norm`、`nemotron_layer_norm`、`fused_add_nemotron_layer_norm`),但 vLLM 中没有任何分发到它们;所有带 bare `nn.LayerNorm` 的模型在 XPU 上都回退到 torch-xpu-ops 的通用 kernel。

## 优化点

新增注册名 `layer_norm` 的 CustomOp(diff 中类名 `LayerNorm`,正文称 `StandardLayerNorm`),作为 bare `nn.LayerNorm` 的 drop-in 替代,按 `forward_native`/`forward_cuda`/`forward_xpu` 分派,与同文件的 `RMSNorm`/`GemmaRMSNorm` 一致;把 `NemotronLayerNorm1P` 从 `nn.LayerNorm` 子类改写为 CustomOp sibling(`nemotron_layer_norm`),其 XPU 路径在有 residual 时也走 fused-residual-add kernel。接入 GPT-2、GPT-J、OPT、SigLIP(PaliGemma vision tower)、Nemotron;并在 `transformers/utils.py` 增加 `replace_layernorm_class`,让 Transformers modeling backend 的通用递归 module-replace 也能命中(覆盖无专用 model file 的 StarCoder2 类模型)。kernel 调用包装成真正的 `torch.ops.vllm.xpu_*` custom op(`_xpu_ops.py` 经 `direct_register_custom_op`),仅由标准 CustomOp `+/-custom_ops` 开关控制、compile-safe;每条分派路径用 `hasattr(torch.ops._C, ...)` 守卫,op 不存在时回退原生 `F.layer_norm`。正文说明当前因尚无含 #577 的 vllm-xpu-kernels release、且 requirements/xpu.txt 的 pin 早于它,实际仍是 no-op。

## 关键改动

```text
vllm/model_executor/layers/layernorm.py: 新增 LayerNorm CustomOp(layer_norm) 与 NemotronLayerNorm1P 改写
vllm/_xpu_ops.py: 用 direct_register_custom_op 包装 xpu_* 融合 SYCL kernel
vllm/model_executor/models/{gpt2,gpt_j,opt,nemotron,siglip}.py: 替换 bare nn.LayerNorm 接入
vllm/model_executor/models/transformers/utils.py: 新增 replace_layernorm_class 供 Transformers backend 递归替换
vllm/models/{deepseek_v32,glm5next,hy_v4}/*/attention.py: 同步替换 LayerNorm 用法
```

关键机制代码:

```python
@CustomOp.register("layer_norm")
class LayerNorm(CustomOp):
    """Standard (mean-centered) LayerNorm.
    Drop-in for a bare `nn.LayerNorm` that dispatches to a fused XPU kernel
    when one is available, and to the native implementation otherwise.
    """
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        ...
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
```

## 性能声明

PR 正文 Test Result:kernel microbenchmark(native `F.layer_norm` vs SYCL,bf16,hidden_size=4096):tokens=1 8.35 → 3.60 us(2.32x)、128 9.63 → 3.70(2.60x)、512 28.20 → 5.90(4.78x)、1024 51.73 → 10.83(4.78x peak)、2048 103.91 → 56.61(1.84x)。E2E(Intel B70,`--enforce-eager`,N=3 median):GPT-2-124M throughput +9.1% / latency -9.5% / serve +5.3%;OPT-125M +9.2% / -1.2% / +9.4%;GPT-J-405M +6.1% / -4.4% / +9.2%;StarCoder2-3B +1.0% / -1.2% / +1.3%;Minitron-8B +0.8% / -0.9% / +0.5%。Caveat:上述数据测自更早一版 dispatch(直接 `torch.ops._C.*` 调用),本 PR 改为经 `torch.ops.vllm.xpu_*` 包装以保证 torch.compile 安全,是同一 kernel call 的薄封装,行为应不变但未在硬件上重跑。

## 验证与回归风险

Test Plan:ruff/format/full pre-commit(含 mypy)、Intel B70 上 kernel 正确性(64 parametrized cases + 2 fused-add cases,对 plain-PyTorch fp32 参考全过)、各模型 `--enforce-eager` E2E serve、以及 `vllm bench serve/throughput/latency` 前后对比。正文明确尚未做:仓库内 correctness test(`tests/kernels/core/test_layernorm.py` 只覆盖 RMSNorm/GemmaRMSNorm)、任何受影响模型的 lm_eval 精度检查。影响路径:XPU 上带 bare `nn.LayerNorm` 的 GPT-2/GPT-J/OPT/SigLIP/Nemotron 及 Transformers backend 模型;当前因缺 vllm-xpu-kernels release 实际 no-op,所有分派有 hasattr 守卫回退 `F.layer_norm`。

## 关联

外部依赖 vllm-project/vllm-xpu-kernels#577(提供融合 SYCL LayerNorm kernel,正文称刚合并、尚无 release 含它);正文未引用其他 vLLM PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
