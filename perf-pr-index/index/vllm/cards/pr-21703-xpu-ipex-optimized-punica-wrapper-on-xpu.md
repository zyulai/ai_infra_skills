# PR #21703: [XPU] IPEX-optimized Punica Wrapper on XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chaojun-zhang |
| 合并 | 2025-07-28 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/21703 |

## 一句话结论

XPU 上用 IPEX Punica 包装器替换 Triton kernel，Llama-2-7b LoRA 吞吐 2082.29→2543.90 tok/s（1.22x）。

## 动机

PR 正文只给出 Test Plan 与两组 benchmark 对比（`XPU_USE_TRITION_KERNEL=0` 的 xpu kernel 与 `=1` 的 triton kernel），未文字说明动机；从 diff 可见 vLLM 的 XPU 平台此前 `get_punica_wrapper()` 返回 `punica_gpu.PunicaWrapperGPU`，本 PR 新增基于 IPEX 的 `punica_xpu.PunicaWrapperXPU` 并切换过去。从 diff 可见，`punica_xpu.py` 文件头引用 Punica 论文（Chen 等，2023，arXiv:2310.18547），`PunicaWrapperXPU` 继承 `PunicaWrapperBase` 并以 `@final` 标注，其 `__init__` 对 `_token_lora_indices`、`_embeddings_indices`、`_sampler_indices_padded` 三个索引张量调用 `torch._dynamo.mark_dynamic` 以配合 torch.compile。

## 优化点

新增 `vllm/lora/ops/ipex_ops/lora_ops.py`，把 `bgmv_shrink`、`bgmv_expand`、`bgmv_expand_slice` 直接转发到 `ipex.llm.functional` 的对应实现；新增 `vllm/lora/punica_wrapper/punica_xpu.py`（`PunicaWrapperXPU`，基于 Punica: Multi-Tenant LoRA Serving 论文，管理 Multi-LoRA 元数据并对 `_token_lora_indices`、`_embeddings_indices`、`_sampler_indices_padded` 调 `torch._dynamo.mark_dynamic`），shrink/expand 通过 `ipex_ops` 调用 IPEX kernel；`vllm/platforms/xpu.py` 的 `get_punica_wrapper()` 由 `punica_gpu.PunicaWrapperGPU` 改为 `punica_xpu.PunicaWrapperXPU`。三个 op 的签名与 vLLM 既有 punica 接口对齐：`bgmv_shrink` 带 `scaling`，`bgmv_expand` 带 `add_inputs`，`bgmv_expand_slice` 额外接收 `slice_offset` 与 `slice_size`；`ipex_ops/__init__.py` 仅做符号导出，`lora_ops.py` 在 `import intel_extension_for_pytorch` 失败时直接抛出。

## 关键改动

```text
vllm/lora/punica_wrapper/punica_xpu.py: 新增 PunicaWrapperXPU（269 行），用 IPEX Punica kernel 管理 Multi-LoRA 元数据。
vllm/lora/ops/ipex_ops/lora_ops.py: 新增 bgmv_shrink / bgmv_expand / bgmv_expand_slice 的 IPEX 转发。
vllm/lora/ops/ipex_ops/__init__.py: 导出上述三个 op。
vllm/platforms/xpu.py: get_punica_wrapper() 切换到 PunicaWrapperXPU。
```

```python
def bgmv_shrink(inputs, lora_a_weights, output_tensor,
                lora_indices_tensor, scaling: float = 1.0) -> None:
    ipex.llm.functional.bgmv_shrink(inputs, lora_a_weights, output_tensor,
                                    lora_indices_tensor, scaling)


def bgmv_expand(inputs, lora_b_weights, output_tensor,
                lora_indices_tensor, add_inputs: bool = True) -> None:
    ipex.llm.functional.bgmv_expand(inputs, lora_b_weights, output_tensor,
                                    lora_indices_tensor, add_inputs)
```

## 性能声明

测试条件：`XPU_USE_TRITION_KERNEL=0 VLLM_USE_V1=1 CCL_ZE_IPC_EXCHANGE=drmfd VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-hf --backend vllm --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --max-loras 1 --max-lora-rank 8 --enable-lora --lora-path "yard1/llama-2-7b-sql-lora-test" --enforce_eager`（XPU；硬件型号与精度 PR 未注明）。xpu kernel + `XPU_USE_TRITION_KERNEL=0`：Throughput 5.29 requests/s、2543.90 total tokens/s、1243.62 output tokens/s（1000 prompts，245995 prompt tokens / 235278 output tokens）；triton kernel + `XPU_USE_TRITION_KERNEL=1`：4.33 requests/s、2082.29 total tokens/s、1017.96 output tokens/s。正文结论：**The XPU kernel achieved a 1.22x throughput improvement (from 2082.29 to 2543.90) over the Triton kernel**。

## 验证与回归风险

PR 正文只给出上述 Test Plan 命令与两组 throughput 输出，未列出单测命令、测试通过数或精度/正确性回归数据，PR 未注明相关测试；两组运行中均出现 `Encountered invalid prefix detokenization error for request ..., resetting decode stream` 警告（分别针对 request 623 与 183）；两者 total num prompt tokens 均为 245995、total num output tokens 均为 235278，输入负载一致。可能影响路径：XPU 平台的 LoRA / Multi-LoRA punica 路径（`get_punica_wrapper` 全局切换到 `PunicaWrapperXPU`），依赖 `intel_extension_for_pytorch`（`lora_ops.py` 中 import 失败会直接 raise）。正文没有给出改动前后 LoRA 输出精度或正确性对照数据，也未列出单元测试，两组 benchmark 的 `Throughput` 行是唯一的定量依据；测试命令带 `--enforce_eager` 与 `VLLM_USE_V1=1`。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
