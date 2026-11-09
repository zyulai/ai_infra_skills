# PR #65498: [Perf][MoE] Skip the combine staging copy in flashinfer_nvlink_one_sided

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vadiklyutiy |
| 合并 | 2026-11-04 |
| 分类 | MoE \| tier: major |
| Label | ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/65498 |

## 一句话结论

去掉 combine staging copy，Qwen3.5-397B/4×B200 端到端 prefill 吞吐 +1.4%~+3.2%（ISL 512–32K）。

## 动机

`flashinfer_nvlink_one_sided` 在每次 combine 前（`moeA2APrepareCombineKernel`）把整个 expert 输出拷进 FlashInfer 的对称 workspace。combine 是 pull-based，各 rank 直接从 owner 的 workspace 读取本 token 的部分结果，因此 expert 输出必须落在该 workspace。而 `FlashInferNVLinkOneSidedPrepareAndFinalize` 未覆写 `allocate_fused_expert_output`，modular kernel 把 vLLM workspace manager 的 buffer 交给 expert，`finalize()` 再把它传给 `combine()`，于是 FlashInfer 额外启动拷贝 kernel 把每行搬进 combine 区。B200 EP4、Qwen3.5-397B shapes 下该拷贝在 16K tokens/rank 时每 MoE 层耗 163 us（约 0.5 GB 读 + 0.5 GB 写 HBM）。AGRS 因已覆写该 hook，从未付这笔拷贝。

## 优化点

让 `allocate_fused_expert_output` 返回 `communication.get_combine_input_buffer(dtype)`，即本 rank combine 区的 `[ep_size * tokens_per_rank, hidden]` 视图，expert 直接写进目标存储；`finalize()` 不变，`NVLinkOneSidedAlltoAll.combine()` 比对 payload 指针后传 `payload_in_workspace=True`，FlashInfer 把拷贝 kernel 缩减为只更新 round flag 的单 CTA。hook 仅在 `TopKWeightAndReduceNoOP`（expert 自行 reduce top-k）、dtype 为 bf16/fp16（combine 区按 16-bit 行定尺寸）、shape 与 device 匹配时生效，否则返回 `None` 保留旧行为。combine 区紧接最后一个 dispatch payload 区且 cacheline 对齐，视图在 dispatch 与 combine 之间有效，hook 只构造张量视图，不引入新的 host-device 同步，也不影响 CUDA graph capture。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py: 覆写 allocate_fused_expert_output，返回 workspace combine 区视图。
tests/distributed/test_mnnvl_alltoall.py: 新增 test_one_sided_expert_output_in_workspace，校验 combine 收到 payload_in_workspace=True 且输出精确。
```

```python
def allocate_fused_expert_output(self, shape, dtype, device,
                                 weight_and_reduce_impl):
    if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
        return None
    if dtype not in (torch.bfloat16, torch.float16):
        return None
    communication = self.all2all_manager.get_communication()
    buffer = communication.get_combine_input_buffer(dtype)
    if tuple(buffer.shape) != tuple(shape) or buffer.device != device:
        return None
    return buffer
```

## 性能声明

微基准（仅通信）：EP4、B200、hidden 4096、top_k 10、512 experts、NVFP4 dispatch、bf16 combine、随机路由，`flashinfer.testing.bench_gpu_time_with_cupti`，cold L2 + CUDA graphs，100 次迭代取最慢 rank 中位数。one-sided dispatch+combine：1K/4K/16K tokens per rank 分别 −5.9 / −43.1 / −160.9 us（表 1024:97.9→92.0、4096:324.5→281.4、16384:1195.3→1034.4 us）；nsys 显示 `moeA2APrepareCombineKernel` 1K 时 13.4 us、16K 时 163.3 us/次。端到端：Qwen3.5-397B-A17B-NVFP4，DP4+EP、4×B200、`--max-num-batched-tokens 32768`、fp8 KV cache、flashinfer_trtllm MoE backend、每点约 10M input tokens、1 output token，ISL 512–32K 的 prefill 吞吐 +1.4% 到 +3.2%（ratio 1.014x–1.032x），每个 ISL 都更快；gsm8k 5-shot 不变（flexible-extract 0.8135±0.0107→0.8211±0.0106，strict-match 0.7930±0.0112→0.7998±0.0110）。

## 验证与回归风险

单测：`tests/distributed/test_mnnvl_alltoall.py` 5 passed、13 skipped（多 GPU 测试因裸机无 `CAP_SYS_PTRACE` 被 `requires_ptrace` 跳过，其中 3 个含新测试在 2 GPU 上直接运行均通过）；`tests/kernels/moe/test_moe_layer.py -k flashinfer_nvlink_one_sided` 2 passed、8 skipped。新增测试通过真实 `finalize()` 确认 `payload_in_workspace=True` 且输出精确。硬件/软件：单 DGX B200 节点（NVSwitch 8 GPU）、FlashInfer 0.7.2、PyTorch 2.13.0+cu130、NCCL 2.29.7，基线为 `main` 的 51c6dfd8c8。回归风险：hook 不适用时（expert 不自行 reduce top-k、输出非 bf16/fp16、shape/device 不匹配）保留 vLLM 自有 buffer 与原有拷贝；combine 区在下一层 dispatch 之后才被覆写，与旧 staging copy 的写入时点一致。

## 关联

#54268（前序）；#59134（相关但不同：为自量化输入的 expert 做 dispatch sizing）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
