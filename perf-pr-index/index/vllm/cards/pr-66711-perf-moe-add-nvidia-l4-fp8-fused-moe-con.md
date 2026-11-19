# PR #66711: [Perf][MoE] Add NVIDIA L4 FP8 fused MoE config for Qwen3-30B-A3B

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shijie-lyu |
| 合并 | 2026-11-17 |
| 分类 | MoE(tier: major) |
| Label | ready, qwen, nvidia |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66711 |

## 一句话结论

为 NVIDIA L4(AWS g6)新增 Qwen3-30B-A3B FP8 fused MoE tuned config(`E=128,N=384,dtype=fp8_w8a8,block_shape=[128,128].json`),消除 "Using default MoE config" 的次优路径:kernel 时间最多降 5.4%,并发 8 服务吞吐 +2.2%、median TPOT -4.0%。

## 动机

`vllm/model_executor/layers/fused_moe/configs/` 里没有 L4 的 fused MoE config,Qwen3-30B-A3B-FP8 在 L4 TP2 上只能用默认启发式并打出 "Performance might be sub-optimal!" 警告。其他同 expert 形状的 Qwen3-30B-A3B FP8 checkpoint(如 Qwen3-Coder-30B-A3B-Instruct-FP8)会加载同一文件;作者只实测了 Qwen3-30B-A3B-FP8。

## 优化点

- 用默认搜索空间与 batch sizes 调优:`python benchmarks/kernels/benchmark_moe.py --model Qwen/Qwen3-30B-A3B-FP8 --tp-size 2 --dtype fp8_w8a8 --tune`,产出一个 171 行的 JSON config。
- 选择性保留 tuned entries:lookup 取最近 key(并列取较小者),tuned 范围为 4-240 tokens(keys 4 到 128 及 224)与 385-528(key 512);1-3、241-384(key 256)、529 及以上(keys 544/768/1024/1536/2048/3072/4096)保留 main 默认。作者逐 batch size 算到 16384 确认其余批大小与 main 完全一致。
- 为什么大批保留默认:tuned config 用 `BLOCK_SIZE_M` 16/32,平均每 expert token 数超过 M tile 后反而慢——320 tokens 慢至 6%、384 慢至 14%、640 慢 2-5%,故在每次运行都仍更快的边界处结束 tuned 范围。
- 1 与 2 tokens:tuned 在 `benchmark_moe.py` 更快但端到端并发 1 decode 慢约 0.7%(benchmark 重放相同输入、活跃 expert 权重约 38 MB 可塞进 L4 48 MB L2),故保留默认;224 key 是 64/96/128/256 的调优结果,让 256 key 能放默认。

## 关键改动

```json
// vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json(新文件)
{
    "triton_version": "3.8.0",
    "4": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 3
    },
    "512": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 5
    }
}
```

(diff 仅含这一个新 JSON 文件;上为摘录,1/2/8/16/24/32/48/64/96/128/224/256/544/768/1024/1536/2048/3072/4096 各 key 见原文件。)

## 性能声明

- Kernel(1x L4,`benchmark_moe.py` 无 `--tune`,ABBA 4 轮):7-240 与 385-528 tokens 的 fused MoE kernel 时间在每次运行中低 1.2-5.4%;4 与 6 tokens 低 0.6-1.6%;5 tokens 持平。
- 服务(c8):吞吐 +2.2%,median TPOT -4.0%(3 次运行的中位数;Qwen3-30B-A3B-FP8,TP2,2x L4,fresh `vllm serve`,server log 确认加载 config)。
- 服务(c32):无可测变化;c1 走 main 的 config(1-token decode、1024-token prefill),只反映噪声底。
- 精度:GSM8K(5-shot,1319 题,temperature 0,`tests/evals/gsm8k/gsm8k_eval.py`)default 0.889 → 本 PR 0.893。
- 环境:NVIDIA L4 24 GB、driver 580.159.04(AWS g6.12xlarge 调优/服务,g6.xlarge kernel A/B);torch 2.14.0+cu132、Triton 3.8.0;vLLM main feed3a4475(config lookup 在该 commit 与本分支间不变)。

## 验证与回归风险

- 第二个 commit 只把 241-384、529-768、1281-1792 tokens 改回 main 的 config;服务与 GSM8K 用第一个 commit 的文件跑——decode c1/8/32 与 1024-token prefill chunk 在两版本用相同 entry。batch 覆盖每个差异区间的两端及 main 默认切换点(5, 9, 17, 65),另含 config 等于 main 的抽查点。
- 回归风险:tuned 范围边界是按"每次运行都更快"选的;在 320/384/640 tokens 等 tuned 区间外的批若被误配会慢 2-14%,而 lookup 保证这些批取 main 默认。PR checklist 未勾选。

## 关联

- 相关文件仅一个新 JSON;benchmark 命令与原始输出在 PR 评论区;正文明示只实测 Qwen3-30B-A3B-FP8,同形状 checkpoint 的效果未测。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
