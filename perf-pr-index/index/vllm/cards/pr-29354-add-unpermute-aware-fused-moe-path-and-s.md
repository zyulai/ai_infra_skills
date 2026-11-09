# PR #29354: Add unpermute-aware fused MoE path and small-batch fallback

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @RunkaiTao |
| 合并 | 2026-01-09 |
| 分类 | MoE |
| Label | performance, ready |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/29354 |

## 一句话结论

为 fused MoE Triton 内核增加绕过 permute/align 的 naive block 分配小批量路径,大 expert 数低 token 场景内核耗时最多降 17.6%。

## 动机

fused MoE 常规路径要求先 `moe_align_block_size` 把 token 按 expert 排序并 pad 到 BLOCK_SIZE_M 的整数倍。对"expert 数很大、每 chunk token 数 * top_k 很小"的工况(decode 小 batch 撞上 Llama-4-Maverick 128 expert、DeepSeek 256 expert 这类模型),排序/pad 产生的数据搬运与无效 block 开销盖过了计算本身,regular kernel 效率低下。

## 优化点

给 `fused_moe_kernel` 增加 `naive_block_assignment` 编译期分支:允许 `sorted_token_ids` 为 None,此时每个 M-block 对应单个 token——`offs_token` 直接由 `pid_m` 构成(首元素为 `pid_m`,其余置为 `num_valid_tokens` 使 token_mask 过滤掉),`expert_ids` 即 `curr_topk_ids.view(-1)`,`num_tokens_post_padded` 填 `topk_ids.numel() * BLOCK_SIZE_M`,从而完全跳过 `moe_align_block_size` 的 sort/permute 步骤。执行层用启发式选择:`expert_map is None 且 tokens_in_chunk * top_k * SPARSITY_FACTOR(=4) <= global_num_experts`,且非 block-quantized(int8/int4 w8a16/w4a16 + block_shape 未实现,排除),不满足则回退原路径。`invoke_fused_moe_triton_kernel`/`dispatch_fused_moe_kernel` 的 assert 与 EM 计算相应适配(`sorted_token_ids is None` 时 `EM = num_tokens * BLOCK_SIZE_M`)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: fused_moe_kernel naive 分支 + 启发式 + sorted_token_ids 可为 None
tests/kernels/moe/test_moe.py: 新增 test_naive_block_assignment_moe 与大 expert/小 topk 配置
```

```python
naive_block_assignment = (
    expert_map is None
    and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
    and not ((use_int8_w8a16 or use_int4_w4a16) and block_shape is not None
             and block_shape[1] > 0)
)
```

## 性能声明

`benchmarks/kernels/benchmark_moe.py`(TP=8,除 GPT-OSS TP=4):Llama-4-Maverick-17B-128E:M=1 33.14→28.19 µs(-17.56%),M=2 -17.43%,M=4 -12.27%,M=8 -7.51%,M=16/24 -4.46%,M=32 -2.82%,M>=48 持平。DeepSeek-R1:M=1 70.69→65.55 µs(-7.84%),M=2 -6.87%,M=4 -3.8%,M>=8 持平。Qwen3-Coder-480B-A35B-FP8:M=1 -8.18%,M=2 -7.28%,M=4 -3.5%。GPT-OSS-120B:M=1 -10.21%,M=2 -7.30%,M=4 -3.03%。精度:Mixtral-8x7B-FP8 gsm8k(100 题)前后均 0.620,输出吞吐 226.787→231.849 tok/s。

## 验证与回归风险

新增 `test_naive_block_assignment_moe`(m∈{1,2}、e∈{128,256}、topk∈{1,2},padding/chunk_size 参数化),对照 torch_moe/iterative_moe 基线;现有 `NUM_EXPERTS_LARGE`、`TOP_KS_SMALL`、`FUSED_MOE_MNK_FACTORS_SMALL_M` 配置扩充常规 MoE 测试矩阵。风险:naive 路径每 block 单 token,大 batch 下退化,由启发式阈值兜底;block-quantized 路径未实现,EP(expert_map 非 None)场景不启用。

## 关联

同分类(MoE):PR #19500(Blackwell FP4 MoE 量化内核优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
