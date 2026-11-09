# PR #61996: [Perf][Sampler] Speed up the small-batch split top-p pipeline

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @abmfy |
| 合并 | 2026-10-17 |
| 分类 | Sampler / Detokenization（tier: major） |
| Label | ready, mrv2 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 9 个文件，核心为 `vllm/v1/sample/ops/topk_topp_triton.py`、`vllm/v1/sample/ops/topk_topp_sampler.py`、`vllm/v1/worker/gpu/sample/{sampler,states}.py`、`vllm/v1/watermarking/gpu_sampler.py`、`vllm/model_executor/models/diffusion_gemma.py` 及 3 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/61996 |

## 一句话结论

小 batch split top-p 流水线解除"每 SM 一个 program"封顶：slice 下限 7168 词条继续切分，无 top-p-only 行时跳过 7 个空 kernel，spec verify top-p 561→219us（2.56x）。

## 动机

#54651 引入的小 batch split top-p 流水线把 grid 封顶在每 SM 一个 program（`S * B <= num_SMs`），该占用度下各 step kernel 是 latency-bound：148-SM 的 B300 在 38 行时 S 掉到 2，38–64 行的一次 top-p 调用耗时约为 37 行的两倍（典型场景：8 请求 x 8 个 spec-decode verify token）。另外当 batch 中没有任何 top-p-only 行时，流水线仍被启动——7 个 kernel 的 program 全部立即退出，纯浪费。

## 优化点

两处机制改动。其一，`_topp_split_count` 签名加入 `vocab_size`，在原条件或"切分后每片仍覆盖至少 `_SPLIT_MIN_SLICE = 7168` 词条"时继续倍增 S（仍为 2 的幂、上限 32）；7168 这个下限由实测选定：约 200K vocab 时 S=32（约 6.3K 词条）比 S=16 慢 3–24%，而 256K–262K 时 S=32 多数更快，故下限须落在 6314 与 8000 之间。S 只影响行划分方式，workspace 大小不变，warmup 已覆盖所有 B，无新增运行时 JIT。其二，`apply_top_k_top_p(_triton)` 新增 `has_topp_only_rows: bool = True`（False 表示没有行同时满足 p<1 且 k>=vocab_size），为 False 时跳过 split 流水线——单体 kernel 本就会 mask 所有行（含 grammar 行）；MRV2 的 `SamplingStates.get_top_k_top_p` 改为返回该标志，用 host 侧 numpy 计算、无 device sync，MRV2 各调用点（verification/logits 处理、非融合随机采样、watermark sampler、diffusion_gemma）透传；MRV1 保持默认 True 仍启动流水线。

## 关键改动

```text
vllm/v1/sample/ops/topk_topp_triton.py: _topp_split_count 加 vocab_size 与 _SPLIT_MIN_SLICE=7168；has_topp_only_rows 为 False 时跳过 _apply_topp_split
vllm/v1/worker/gpu/sample/states.py: get_top_k_top_p 返回 (top_k, top_p, has_topp_only_rows)，numpy host 侧计算
vllm/v1/sample/ops/topk_topp_sampler.py: apply_top_k_top_p 透传 has_topp_only_rows
vllm/v1/watermarking/gpu_sampler.py、diffusion_gemma.py: 调用点适配
tests/v1/sample/test_topk_topp_sampler.py 等: 新增 skip/hint/大 vocab 混合批次测试
```

```python
while splits < _SPLIT_MAX_SPLITS and (
    splits * 2 * batch_size <= num_sm
    or splits * 2 * _SPLIT_MIN_SLICE <= vocab_size
):
    splits *= 2
```

## 性能声明

微基准：CUDA graph 每 variant 一图、100 次 replay 取中位、L2 热、p=0.95/k=50（`apply_top_k_top_p_triton`）。Main→PR：top-p-only 行 B300 1.22–2.16x、H200 1.05–1.90x；top-k 与 top-p-only 混合（B≥32、V≥120K）1.43–2.70x（单条 top-p-only 行最高 3.7x@V=151936、4.4x@V=262144，B300）；全 top-k 批次 1.00–1.21x（MRV2 不再启动流水线）；最差 0.97x 在无变化配置的 A/A 噪声内。真实 MiMo-V2.6 verify logits（B300，top-p only）：B=38 561.2→219.0us（2.56x）、B=64 565.0→317.8us（1.78x）。E2E（B300，Triton sampler 路径）：Llama-3-8B-Instruct seeded top-p c=48 top-p 573→166us/步、median ITL −5.1%；MiMo-V2.6-Pro TP4 DFlash K=7 c=8 top-p 558→301us/步、decode 步 22.7ms 中 GPU busy（除 MoE GEMM）降 0.27ms。精度：GSM8K 5-shot Llama-3-8B（T=0.6, top_p=0.9, c=64, 6 run）main 73.25±1.11 vs PR 73.82±0.96。

## 验证与回归风险

测试：B300 跑 sample 相关 6 个测试文件 287 passed/1 skipped，watermarking 191 passed/3 skipped；H200 `test_topk_topp_sampler.py` 140 passed；CI 覆盖 H200 MIG 与 MI355。新增：`test_skip_split_without_topp_only_rows`（hint=False 时不启动 split 且结果 bitwise 一致，含 grammar-masked 行）、`test_has_topp_only_rows`（含禁用 top-k 的取值）、B=48 参数（128–152-SM GPU 上唯一测到 main 用 S=2 的规模）、大 vocab 混合批次测试。输出变化：kept set 仅在 p 边界 tie 处不同（真实 bf16 logits 上均为精确 tie；合成 fp32 logits 41424 行评测中 3671 行不同、kept mass 差 ≤1.1e-4，源于 fp32 求和顺序）。风险面：ROCm 与 <132 SM GPU 未实测（作者无硬件），MI300X 上 V≈152K 从 B=20 起 S 变 16；无 hint 的调用方全 top-k 批次 0.90–0.98x（+2~23us）；mask kernel 的 tie 计数每 8192 词条重启为 PR 之前已存在的 bug。

## 关联

前序：#54651（split top-p 流水线本体）。相关：#55414（B 分桶，作者认为可组合）、#56154（draft，携带同款 dispatcher）、#58231/#59540/#59804/#60104（同文件其他改动）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
