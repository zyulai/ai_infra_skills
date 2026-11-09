# PR #2979: Optimized fused MoE Kernel, take 2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pcmoritz |
| 合并 | 2024-02-26 |
| 分类 | MoE (tier: major) |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/2979 |

## 一句话结论

更激进的参数搜索调优 fused MoE kernel,H100 TP2 下各 qps 的 ITL 由 23.3~60.8 ms 降到 11.6~32.9 ms。

## 动机

本 PR 替换 #2913。正文称通过更激进的 parameter search,即便在小 batch size 下也超过了 TensorRT 的 MoE kernel,"非常幸运,因为大幅降低了复杂度"(即不再需要针对不同 batch 混用 TensorRT kernel)。此前 main 分支的 fused MoE kernel 未经调优。

## 优化点

以 autotuning 搜索 Triton fused MoE 的 tile / warp 配置并固化调优结果:新增 `benchmarks/kernels/benchmark_mixtral_moe.py` 做 grid search——按 batch size 选 `BLOCK_SIZE_M` 候选(bs<=16 用 [16],随 bs 增大逐步加入 32/64/128/256),再遍历 `BLOCK_SIZE_N` [32,64,128,256]、`BLOCK_SIZE_K` [64,128,256]、`GROUP_SIZE_M` [1,16,32,64]、`num_warps` [4,8](`num_stages=4`),每个 config warmup 后测时,选 `best_time_us` 最小的写进 `/tmp/config.jsonl`。调优产物以 E=8,N=3584(A100)/ E=8,N=7168(H100)的 JSON 配置文件形式放入 `vllm/model_executor/layers/fused_moe/configs/`,`fused_moe.py` 加载这些配置;正文强调代码结构使"往 `fused_moe/configs` 目录丢一个新的 .json 文件即可支持不同配置"。

## 关键改动

```text
benchmarks/kernels/benchmark_mixtral_moe.py: 新增 grid-search 调优脚本,遍历 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps 并把最优 config 写 /tmp/config.jsonl
vllm/model_executor/layers/fused_moe/configs/*.json, configs/README: H100 / A100 的 E=8 调优配置与说明,新增配置只需放一个 json
vllm/model_executor/layers/fused_moe.py, fused_moe/__init__.py: fused MoE kernel 加载调优配置
setup.py: 打包/配置相关改动
```

关键代码(grid search 的配置空间):

```python
    for block_size_n in [32, 64, 128, 256]:
        for block_size_m in BLOCK_SIZES_M:
            for block_size_k in [64, 128, 256]:
                for group_size_m in [1, 16, 32, 64]:
                    for num_warps in [4, 8]:
                        configs.append({
                            "BLOCK_SIZE_M": block_size_m,
                            "BLOCK_SIZE_N": block_size_n,
                            "BLOCK_SIZE_K": block_size_k,
                            "GROUP_SIZE_M": group_size_m,
                            "num_warps": num_warps,
                            "num_stages": 4,
                        })
```

## 性能声明

正文给出 "results on this PR on H100 with TP2" 的 ITL(ms)对比(PR 正文未注明模型、精度与 workload 细节):本 PR qps=1 → 11.6、qps=2 → 12.6、qps=4 → 23.6、qps=6 → 32.9;current main(untuned fused MoE kernel)qps=1 → 23.3、qps=2 → 25.4、qps=4 → 43.0、qps=6 → 60.8;only using the TensorRT Moe kernels qps=1 → 18.1、qps=2 → 23.8、qps=4 → 48.1、qps=6 → 90.8。正文未给出吞吐百分比提升数字,也未提供精度结果。注:diff 中 `benchmark_mixtral_moe.py` 的 grid search 使用 Mixtral 形状(d_model=4096、num_total_experts=8、top_k=2、tp_size=2、model_intermediate_size=14336、num_layers=32),但 PR 正文未声明该 benchmark 的模型名。

## 验证与回归风险

PR 正文仅列出上述三组 ITL 对比与调优脚本,未给出单元测试、pytest 命令或精度(eval)结果。风险面:调优配置按 device_name(`E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3` / `E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB`)固化,影响对应硬件上的 fused MoE 路径;正文称更激进的搜索在小 batch 也胜过 TensorRT、从而降低混用复杂度,未列其他模型或硬件回归测试。

## 关联

PR 正文明确:本 PR 替换 https://github.com/vllm-project/vllm/pull/2913(#2913);Co-authored-by: Cade Daniel。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
