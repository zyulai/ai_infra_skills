# PR #20520: [NPU]TP Communications compression For Qwen3 models for NPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @egvenediktov |
| 合并 | 2026-05-02 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | documentation, quant, npu, run-ci |
| Diff | diffstat 缺失(素材未统计行数);正文列出 9 个文件,diff 覆盖 server_args、linear、communicator、distributed 与模型文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/20520 |

## 一句话结论

NPU 上把 TP 通信(attention o_proj / MLP down_proj 后的 all-reduce)做 INT8 压缩,prefill 密集基准平均提速 5%。

## 动机

TP 在 attention 的 o_proj 之后与 MLP 的 down_proj 之后引入设备间通信。为降低通信延迟,可在发送前量化、通信完成后立即反量化。正文附有 FP 与压缩情形的 TP 通信 profiling 对比图。

## 优化点

新增 `--enable-quant-communications` server 参数(默认关,校验 tp_size != 1 且 device 为 NPU)。核心在 `NpuCommunicator.quant_all_reduce`:用 `torch_npu.npu_dynamic_quant` 做 per-token 对称 INT8 量化,把一次 all-reduce 拆成"低精度 all-gather + 全精度 reduce"——值张量与 scale 分别 `all_gather_into_tensor`,反量化后 reshape 求 sum。接入三处:RowParallelLinear.forward 接收 `forward_batch` 并按条件走 `tensor_model_parallel_quant_all_reduce`;`communicator.py` 的 residual add + all_reduce 路径同理走 `attention_tensor_model_parallel_quant_all_reduce`;qwen2/qwen3 模型的 MLP 层把 `forward_batch` 传到 down_proj。条件限定为非 decode/idle(即仅 prefill)且 flag 开启,其余路径保持原 all_reduce。

## 关键改动

```text
python/sglang/srt/distributed/device_communicators/npu_communicator.py: quant_all_reduce——npu_dynamic_quant INT8 + all_gather + 全精度 reduce
python/sglang/srt/distributed/parallel_state.py / communication_op.py: quant_all_reduce 接口(NPU only)
python/sglang/srt/layers/linear.py / communicator.py: prefill 时按 flag 切换量化通信
python/sglang/srt/models/{qwen2,qwen3}.py: ForwardBatch 传递到 down_proj
python/sglang/srt/server_args.py: --enable-quant-communications 与校验
```

```python
def quant_all_reduce(self, x):
    x_q, scale = npu_dynamic_quant(x, dst_type=torch.int8)
    dist.all_gather_into_tensor(output_tensor, x_q, group=self.group)
    dist.all_gather_into_tensor(output_scale, scale, group=self.group)
    output_tensor = output_tensor.to(x.dtype) * output_scale.unsqueeze(-1).to(x.dtype)
    return output_tensor.reshape((world_size,) + input_size).sum(dim=0)
```

## 性能声明

Qwen3-32B,A2 服务器(Ascend NPU,TP2),prefill 密集基准(`--chunked-prefill-size -1 --disable-radix-cache`,加速 = FP 时间 / INT8 时间或吞吐比):正文文字仅给出"平均 5% 性能提升";各并发/序列长度的具体数字在 PR 正文的截图中,文本素材未包含。

## 验证与回归风险

准确性:Qwen3-32B TP2 上对比 BoolQ(3270 题)、C-Eval(1346 题)、HellaSwag(50000 题),baseline 与量化通信的精度截图对比,正文称"平均无退化"(具体数值在截图中,文本素材未包含)。新增 2 个注册 NPU nightly 测试:Qwen3-8B 与 Llama-2-7B 开启 `--enable-quant-communications` 跑 GSM8K,精度阈值分别 0.85 与 0.18(suite nightly-2-npu-a3)。风险:量化仅作用于 prefill 路径,decode 通信不变;INT8 量化属有损压缩,精度证据依赖上述基准;`forward_batch` 透传改动触及 RowParallelLinear.forward 签名与 LoRA 层 forward 签名,其他模型需传参才能享受该路径(不传则行为不变)。

## 关联

无(其他同分类 PR 编号素材未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
