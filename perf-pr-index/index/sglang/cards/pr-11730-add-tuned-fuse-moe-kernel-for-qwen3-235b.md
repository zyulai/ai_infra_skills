# PR #11730: add tuned fuse moe kernel for qwen3 235b fp8 on h200

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @pdasgup |
| 合并 | 2025-10-17 |
| 分类 | Speculative Decoding (tier: major,classify.py 给出;实际改动为 fused MoE Triton 调优配置) |
| Label | (none) |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,仅新增 1 个 JSON 配置文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/11730 |

## 一句话结论

为 Qwen3-235B-A22B fp8_w8a8 在 H200 上新增调优后的 fused_moe_triton 配置,intertoken latency 提升约 8%。

## 动机

fused_moe_triton kernel 的 tile/warp/stage 配置按 (专家数, N, 设备, dtype) 维度存放在配置 JSON 中;Qwen/Qwen3-235B-A22B-Instruct-2507 的 fp8_w8a8 形状(E=128, N=192)此前没有 H200 的调优结果,运行时只能落到默认启发式配置,MoE GEMM 在该硬件上未达最优。PR 用官方 tuning 脚本离线搜参后落盘一份 H200 专用配置。

## 优化点

不改动任何 kernel 代码,纯新增调优产物:`E=128,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`,覆盖 batch size 1 到 4096 共 18 档,每档给出 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages。调优结果与默认配置的差异明显:小 batch(1-256)从默认 BLOCK_SIZE_M=64/128 降到 16,降低 padding 浪费;num_stages 从 4 降为 3 换取更好流水;大 batch(2048+)把 BLOCK_SIZE_K 从 128 降到 64 等。运行时按 batch size 查表选用,无需预热。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8.json: 新增,唯一改动文件,triton 3.4.0 下 Qwen3-235B fp8_w8a8 在 H200 的 18 档 batch size 调优配置。
```

小 batch 配置示例:

```json
"1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
       "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}
```

## 性能声明

单 kernel 对比(`tuning_fused_moe_triton.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 --tp-size 8 --dtype fp8_w8a8`):batch 1 从 31.70 us 降到 27.84 us,batch 64 从 132.86 us 降到 97.69 us,batch 4096 从 536.76 us 降到 496.94 us。E2E serving benchmark(100 请求,平均输入 1138.95 token / 输出约 1001 token):TPOT mean 7.50 -> 7.10 ms,output token throughput 130.77 -> 137.82 tok/s,request throughput 0.131 -> 0.138 req/s,总时长 765.28 s -> 727.15 s;Mean Accept Length 2.15 -> 2.18(表明 E2E 测试启用了投机解码)。作者总结:inter token latency 约 8% improvement。E2E 测试的模型/硬件未在正文逐项注明(按调优与配置文件推断为 H200 + tp8);Accuracy Tests 一节为空。

## 验证与回归风险

验证依赖 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` 的 kernel 级对比与上述 E2E 结果;未新增单元测试(checklist 未勾选)。回归风险很低:配置文件按 device_name/dtype/E/N 精确匹配,只影响 triton 3.4.0 + H200 + Qwen3-235B-A22B fp8_w8a8 的 MoE 路径;其他模型、其他 GPU、其他 triton 版本不会命中该文件。潜在问题是配置对 triton 版本目录(triton_3_4_0)敏感,升级 triton 后需重新调优。

## 关联

同区域(fused MoE Triton 调优)还有 PR #18782(优化 fused moe triton kernel)等;本批次内无直接前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
