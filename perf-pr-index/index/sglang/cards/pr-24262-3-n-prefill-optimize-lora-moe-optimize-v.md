# PR #24262: (3/n - prefill optimize)[LoRA][MoE] Optimize virtual experts: remove CPU-GPU sync & multi-block CUDA JIT histogram

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yushengsu-thu |
| 合并 | 2026-05-11 |
| 分类 | MoE(major) |
| Label | high priority, lora, run-ci, jit-kernel |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 3 个文件(JIT kernel、virtual_experts.py、测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/24262 |

## 一句话结论

LoRA virtual experts 路径的 `_align_block_size` 从 torch.compile fallback 换成 CUDA JIT kernel,微基准提速 8x 以上,并间接消除 TP AllReduce 因 CPU stall 产生的空转(-68%)。

## 动机

LoRA+MoE virtual experts 场景(num_moe_experts × max_loras 可远超 1024 个"专家")此前在 `num_experts > 1024` 时走 `_align_block_size_torch`:torch.compile 版 fallback 含 15+ 次独立 kernel launch 与 `torch.argsort`,CPU 侧开销巨大。在 prefill-heavy 多 TP 工作负载下,每层的 CPU stall(约 3.8s 的 Torch-Compiled Region + 0.8s 的 aten::sort)会让其他 TP rank 的 NCCL AllReduce kernel 在内核里 spin-wait 空等,拖垮整体 prefill。

## 优化点

新增 `_align_block_size_jit`,复用 `moe_align_kernel.cu` 的 v2 kernel(原来只支持 ≤4096 experts,本 PR 将 EXPERTS_PER_THREAD 上限从 4 提到 8,支持至多 8192 个 expert bucket,即 8191 个真实 virtual experts),把 histogram、padded prefix-sum、expert_ids 分配、token scatter 压缩到 2-3 次 CUDA kernel launch。JIT kernel 采用 +1 offset 约定(EP sentinel -1 映射到 bucket 0)。还做了工程优化:4 个输出子缓冲合并为单次 `torch.empty` 分配再切片,避免多次分配的 Python 开销,并按 VEC_SIZE=4 对齐防止向量化写越界污染相邻子缓冲。原 torch.compile 路径保留为 AMD/ROCm 或 JIT 加载失败时的 fallback。

## 关键改动

```text
python/sglang/jit_kernel/csrc/moe/moe_align_kernel.cu: v2 kernel expert 上限 4096→8192,新增 EXPERTS_PER_THREAD=8 分支
python/sglang/srt/lora/triton_ops/virtual_experts.py: 新增 _align_block_size_jit + 单次分配切片;_align_block_size_large 改为 JIT 优先、异常时回退 torch
test/registered/lora/test_virtual_experts_kernels.py: sentinel bucket 测试参数化为 torch/JIT 两条路径
```

关键机制片段(JIT 路径 dispatch):

```python
def _align_block_size_large(topk_ids, block_size, num_experts):
    try:
        return _align_block_size_jit(topk_ids, block_size, num_experts)
    except Exception:
        return _align_block_size_torch(topk_ids, block_size, num_experts)
```

## 性能声明

- 微基准 `_align_block_size`(50 repeats,JIT vs PyTorch):8E×128L/64 tokens 0.048ms vs 0.384ms(8.0x);8E×256L/256 tokens 8.5x;64E×64L/256 tokens 8.6x;8E×256L/1 token(decode)8.8x。
- AllReduce 副作用(Qwen3-30B-A3B + LoRA,TP=4,GB300,input=65536/output=32,10 prompts,`--max-concurrency 4`):`ncclDevKernel_AllReduce_Sum_bf16` 总时长 6,778ms→约 2,192ms,降 68%。PR 自述该归因来自 pre/post trace 对比 + stall source 匹配,post 侧用 TP-0×4 近似,未做逐 PR 增量测量。

## 验证与回归风险

精度验证:3 个模型带 LoRA adapter,TP=4 GB300,对比 trainer 与 orig_sampler 的 KL 均通过(Qwen3-30B-A3B-Instruct-2507 KL=0.00392;Qwen3-VL-30B-A3B-Instruct KL=0.00072;gpt-oss-20b KL=0.00094)。测试方面更新 `test/registered/lora/test_virtual_experts_kernels.py`,为 torch 与 JIT 两条路径共享 sentinel bucket 回归测试(-1/越界 id 不得 OOB 索引、不得分给真实 expert)。风险:JIT 路径 hard cap 8191 experts,超出仍会异常后走慢速 fallback;AMD/ROCm 无 JIT,只能保留原性能。

## 关联

前序 PR #24246(移除 `_add_lora_gate_up_delta` 中 `.any().item()` CPU-GPU sync,本 PR 基于 #24246);本 PR 为 prefill optimize 系列第 3 篇(标题 3/n)。与本卡同属 LoRA virtual experts 区域的 PR 暂无其他收录。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
