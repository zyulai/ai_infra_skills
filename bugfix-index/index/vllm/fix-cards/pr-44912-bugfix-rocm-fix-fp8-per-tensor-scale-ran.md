# PR #44912: [Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion failure

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44912](https://github.com/vllm-project/vllm/pull/44912) |
| 作者 | nehmathe2 |
| 合并日期 | 2026-06-17 |
| labels | bug, ready, rocm |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
FP8 per-tensor 下权重 scale 是 0-D、AITER 激活 scale 是 1-D,torch.compile 降级 `aten._scaled_mm` 时 rank 断言失败、EngineCore 启动即崩;把 0-D scale 归一为 1-D。

## 问题现象
ROCm 上同时启用 AITER(`VLLM_ROCM_USE_AITER=1`)并禁用 skinny GEMM(`VLLM_ROCM_USE_SKINNY_GEMM=0`),跑静态 FP8 per-tensor 量化模型时,torch.compile 抛出:

```
torch._inductor.exc.InductorError: LoweringException: AssertionError:
  target: aten._scaled_mm.default
  File "torch/_inductor/template_heuristics/triton.py", line 1879, in adjust_kernel_inputs
    assert len(scale_a.get_size()) == len(scale_b.get_size())
```

EngineCore 在基准运行期间启动失败。修复后模型加载、编译、serving 正常,基准完成。

## 根因
正文披露:`requantize_with_max_scale` 返回 0-D 的权重 scale,而 AITER 的 `_rocm_aiter_per_tensor_quant_fake` 总是返回 1-D 的激活 scale;当 torch.compile 降级 `aten._scaled_mm` 时,断言 `scale_a` 与 `scale_b` 的 rank 匹配,断言失败导致启动崩溃。

## 修复方案
改 `vllm/model_executor/kernels/linear/scaled_mm/pytorch.py` 的 `apply_scaled_mm`:调用 `torch._scaled_mm` 前把 0-D scale 归一为 1-D,保证编译期 rank 一致:

```python
# torch._scaled_mm under torch.compile does not support 0-D scales
if As.dim() == 0:
    As = As.view(1)
if Bs.dim() == 0:
    Bs = Bs.view(1)
```

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅改动 `apply_scaled_mm` 入口,对 0-D scale 做 `view(1)` 视图变换(无数据拷贝);非 0-D scale 与非 torch.compile 路径行为不变。受益场景为 ROCm + AITER + 静态 FP8 per-tensor + torch.compile。验证在 MI355X 上以 `Llama-3.1-8B-Instruct-FP8-KV` 完成(启动命令含 `AMDGCN_USE_BUFFER_OPS=1`、`--kv-cache-dtype fp8`、`--async-scheduling`、`--attention-backend ROCM_AITER_FA`);其他平台/组合未在正文列出。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
