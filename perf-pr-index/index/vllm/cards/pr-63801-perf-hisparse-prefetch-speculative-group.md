# PR #63801: [Perf][HiSparse] Prefetch speculative groups with bounded copy workers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhenwenqi2024 |
| 合并 | 2026-11-03 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, performance, ready, ci/build, hisparse |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `csrc/hisparse_kernels.cu`、`vllm/v1/hisparse/runtime.py`、测试与基准脚本等 7 文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/63801 |

## 一句话结论

把 HiSparse 共享索引组预取扩展到 MTP 等多 token 验证 batch,并把 compact gather 限制在 64 个 block 内,8K 输出吞吐 +2.35-3.00%。

## 动机

HiSparse 已有单 token decode 的共享索引组预取,但投机验证(MTP=3 时 32 请求 × 4 行 = 128 行)会让 batch 膨胀:一是组预取资格判断把验证 batch 排除在外,follower 层的历史 KV 无法提前 stage;二是 CUDA compact gather 每行启动一个 block,验证使 gather 网格膨胀,挤占计算占用。相关背景:#60072;与 #62592(平台接口)分开,与 #60103 组合(其 slot-invalidation 融合等不在此重复)。

## 优化点

两处机制。其一,新增 `is_hisparse_group_prefetch_batch`:`num_actual_tokens <= max_swap_rows`、全为 decode token 的纯 decode 验证 batch 也获得 `prefetch_decode_group` 资格——索引层一次性解析 top-k 后把各 follower 层缺失的历史行排入组 copy stream,每层有独立 completion event,后层拷贝可与前层计算重叠;prefill/混合 batch 保持 per-layer staging。当前验证 token 的 resident KV 写入与 host mirror 策略不变,被拒绝的 draft token 不会发布为持久 host KV,且不新增任何开关或 CPU readback。其二,`hisparse_gather_compact` 的 grid 从 `(num_rows, num_chunks)` 改为 `(min(num_rows, 64), num_chunks)`,kernel 内层用 `row += gridDim.x` 循环覆盖超出 64 的行,≤64 行的 batch 保持原有列并行。

## 关键改动

```text
csrc/hisparse_kernels.cu: hisparse_gather_compact_kernel 增加 num_rows
  循环;launch grid 上限 64 block
vllm/v1/hisparse/runtime.py: is_hisparse_group_prefetch_batch 与
  prefetch_decode_group 状态;_resolve_and_stage_group 扩展到 followers
docs/design/hisparse.md: 设计文档新增小节
tests/v1/attention/test_sparse_mla_backends.py 与 tests/v1/worker/test_utils.py:
  bounded gather、验证预取、资格判定测试
benchmarks/kernels/benchmark_hisparse_gather.py: 新增 gather 基准
```

## 性能声明

8×H200,GLM-5.2-FP8,TP=8,MTP=3,V2 runner,kv_cache_dtype=fp8_ds_mla,32 GiB GPU KV/rank,128 GiB host pool,prefix caching 关闭,冻结 main `a1cfd2c1` 背靠背对比:65,536 输入、32 并发,4K/8K 强制输出。8K 输出:吞吐 780.25 → 802.42 tok/s(+2.84%,同进程重复 +2.35%,restart +3.00%),TPOT 26.80 → 25.35 ms(-5.41% 至 -4.10%);4K 输出:吞吐 +1.41% 至 +1.57%,TPOT -2.78% 至 -3.33%;TTFT 基本不变(约 107.7 s,含排队);两 arm 零抢占。24 GiB Torch-profiler 诊断(26 请求/104 验证行):每步 GPU 纯 decode 包络 63.06 → 54.87 ms(约 -13.0%,作者明确声明这不是端到端 13% 提速),非重叠 gather 时间 14.99 → 5.32 ms,重叠率 0% → 67.44%(总 gather 时间反而 14.99 → 16.35 ms)。GSM8K 全集回归(1319 题,thinking max):97.4223% → 97.1948%(-0.2274pp,差异在输出预算失败间摆动),MTP 接受率 85.83% → 84.94%,作者不声称精度或接受率改善。

## 验证与回归风险

验证:rebase 后 CPU 契约测试 10 passed;`tests/v1/worker/test_utils.py -k hisparse` main 45 → PR 53 passed;设备级 supplemental 测试(eager + CUDA Graph、CPU 逐字节 golden、656 字节行、top-k 2048)通过,Compute Sanitizer memcheck 0 错误;pre-commit/clang-format 通过。未完成:`tests/v1/e2e/general/test_hisparse.py` 标准套件、long-history offload miss 下真实权重精度、多并发/MTP 开关扫描。风险:64 block 上限用拷贝带宽换计算重叠,块数限制并不预留特定 SM;性能数字未在新 main 基座上重跑。

## 关联

同区域:#60072(组预取背景)、#60103(launch/resolver 开销与融合,与本 PR 互补)、#61263(residency hash 探测)、#62592(平台接口)。同分类投机解码:#23569、#25916、#24505。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
