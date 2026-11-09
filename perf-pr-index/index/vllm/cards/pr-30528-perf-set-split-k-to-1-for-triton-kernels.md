# PR #30528: [Perf] Set split_k to 1 for triton_kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2025-12-12 |
| 分类 | Attention / Kernel |
| Label | performance, ready, gpt-oss, nvidia |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/30528 |

## 一句话结论

Hopper 上强制 triton_kernels matmul_ogs 的 split_k=1,消除小 batch 走 FP32 的劣化,gpt-oss-120b bs=8 吞吐 +38.9%。

## 动机

issue #28894 报告小 batch 下 triton_kernels MoE matmul 比 marlin 慢。根因在 triton 的 opt_flags:`split_k = max(n_sms // grid_size, 1)`,当 m(batch)小于约 32 时 split_k 变成大于 1,而 split_k>1 时 matmul_ogs 会把输出 dtype 退回 torch.float32(见 triton v3.5.1 matmul_ogs.py)。于是 decode 阶段跑的是 `_matmul_ogs_NNT_fp32xbf16xmxfp4_16x256x128x5`(split_k=5、FP32 输出):FP32 精度引入额外开销,且 split-k 分割对本来就很小的 matmul 没有意义。大 batch 时 grid_size 大,split_k 本来就是 1(Blackwell sm100 因 is_persistent=True 也已是 1),只有 Hopper(sm90)小 batch 受影响。

## 优化点

在 mxfp4 权重的 swizzle/preprocess 处按设备能力分别设置 triton_kernels 的 opt_flags 约束:sm90 上新增 `split_k: 1` 约束,强制任意 batch 都用 bf16 输出、单 pass 的 matmul_ogs;sm100 维持原有 `is_persistent: True` + `epilogue_subtile: 1`(其本身即隐含 split_k=1)。改动只影响 triton_kernels 的优化标志选择,不动内核本身;decode 阶段实际调用的内核由 `_matmul_ogs_NNT_fp32xbf16xmxfp4_16x256x128x5` 变为 `_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1` 及其 `_swiglu` 变体(输出 bf16、split_k=1)。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/mxfp4_utils.py: 按 capability(90/100)设置 triton_kernels opt_flags 约束,sm90 强制 split_k=1
```

关键片段:

```python
if current_platform.is_cuda():
    if current_platform.is_device_capability(90):
        constraints = {
            "split_k": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif current_platform.is_device_capability(100):
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
```

## 性能声明

openai/gpt-oss-120b,TP1,sharegpt 数据集 1000 prompts,ignore-eos,vllm bench serve(serve 端 max-num-seqs 与并发一致;benchmark 所用 GPU 正文未注明,举例处提到 H200 n_sms=132):

- bs=8(max-num-seqs 8,max-concurrency 8):Output token throughput 575.28 → 799.32 tok/s(+38.9%);Request throughput 2.89 → 4.02 req/s;Mean TPOT 13.65 → 9.80 ms;Mean TTFT 42.37 → 34.82 ms;Total Token throughput 1197.61 → 1664.02 tok/s。
- bs=256:max-num-seqs 256 下无回退(Request throughput 163.38 → 164.09,TPOT 220.68 → 214.58 ms,P99 TTFT 1409.40 → 2229.85 ms 有波动)。

## 验证与回归风险

精度:gpt_oss.evals GPQA(low reasoning,n-threads 200):bs=8 下 main 0.6591 → PR 0.6521,作者归因于输出精度从 FP32 变为 BF16 的小幅下降;bs=256 下 0.6503 → 0.6591,无回退。回归风险:sm90 上所有经 triton_kernels/mxfp4 路径(如 gpt-oss)的 matmul_ogs 不再使用 split-k;对中等 batch(m≈32 附近)的 shape,原 split-k+FP32 与新单 pass+bf16 的相对性能与精度权衡由作者按 e2e 与 GPQA 验证,极端长序列/其他 mxfp4 模型未单独报告。

## 关联

关联 issue:vllm-project/vllm#28894(triton 小 batch 比 marlin 慢的报告)。同区域:marlin MoE 内核(对比基线)、mxfp4/gpt-oss 量化路径;pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
