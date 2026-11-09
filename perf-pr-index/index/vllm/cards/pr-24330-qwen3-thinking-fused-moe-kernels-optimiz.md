# PR #24330: QWEN3 Thinking Fused MoE kernels Optimization configs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @samanamp |
| 合并 | 2025-09-07 |
| 分类 | MoE |
| Label | performance, ready, qwen |
| Diff | diffstat 缺失(pack 未提供);diff 可用,仅涉及 4 个 fused MoE 调优配置 JSON(B200/GB200,E=128,N=384/768) |
| 链接 | https://github.com/vllm-project/vllm/pull/24330 |

## 一句话结论

为 Qwen3-235B-A22B-Thinking-2507-FP8 的 fused MoE Triton 内核补充/更新 B200、GB200 调优配置,端到端 Output 吞吐 +13.7%、P99 TPOT -17%。

## 动机

Qwen3 Thinking(Qwen3-235B-A22B-Thinking-2507-FP8,E=128 专家)在 B200/GB200 上以 fp8_w8a8、block_shape=[128,128] 运行时,缺 N=384 场景的调优配置文件、N=768 场景的既有配置也未针对这些 GPU 调过,Trirton fused MoE 内核只能落到次优的 block/warp/stage 组合,长上下文大并发 decode 下 TPOT 偏高且尾延迟(P99 ITL 136.71 ms)严重。

## 优化点

纯配置文件改动,不含内核代码:新增 `E=128,N=384` 与 `E=128,N=768` 在 NVIDIA_B200 / NVIDIA_GB200 上的 `fp8_w8a8,block_shape=[128,128]` 配置(共 3 新增 1 更新)。每个 JSON 按 batch size(1 到 4096)给出 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages` 的离线调优结果:小 batch 用 BLOCK_SIZE_M=16、GROUP_SIZE_M=1,大 batch(512+)切到 BLOCK_SIZE_M=64、更大 GROUP_SIZE_M,并为不同档位选择不同 BLOCK_SIZE_K(128/256)与 num_stages(3-5)。fused MoE 运行时按 (E, N, dtype, block_shape, device) 查表选取最优 kernel 配置,机制不变。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增
vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增
vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json: 更新
vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增
```

```json
"512": {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 16,
    "num_warps": 4,
    "num_stages": 4
}
```

## 性能声明

E2E serving:Qwen/Qwen3-235B-A22B-Thinking-2507-FP8,TP4、`--enable-expert-parallel`、max-model-len 65000,random input 7500 / output 7500、并发 64、128 prompts:

- Output token throughput:1666.03 → 1895.36 tok/s(**+13.7%**)
- Median TPOT:38.16 → 33.53 ms(**-12%**);P99 TPOT:40.42 → 33.65 ms(**-17%**)
- P99 ITL:136.71 → 33.69 ms;Mean TTFT 基本持平(6815.24 → 6768.80 ms)

另有 `benchmark_moe.py --dtype fp8_w8a8 --tp 4` 内核级单测截图(PR 正文为图片,无文字数字)。

## 验证与回归风险

验证即上述 E2E 与内核级 benchmark,未新增自动化测试(配置文件本身无单测)。风险面:仅影响命中 (E=128, N=384/768, fp8_w8a8, block_shape=[128,128], B200/GB200) 查表的模型;N=768 B200 既有配置被覆盖,若其他模型恰好命中该键,其行为也会变化(PR 未列出受影响的其他模型)。

## 关联

同 MoE 区域:#39510(TRTLLM GEN NVFP4 MoE 支持 256 对齐 hidden_dim)、#31722(Qwen3-Next GDN decode 提速)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
