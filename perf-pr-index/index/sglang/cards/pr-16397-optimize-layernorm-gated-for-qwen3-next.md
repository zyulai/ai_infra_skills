# PR #16397: Optimize layernorm_gated for Qwen3-Next

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-01-10 |
| 分类 | Attention / Kernel |
| Label | sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 5 个文件,其中 2 个新文件(benchmark 与 test) |
| 链接 | https://github.com/sgl-project/sglang/pull/16397 |

## 一句话结论

Triton layernorm_gated kernel 由一行一 block 改为多行一 block 并行,长序列下 kernel 快 2 倍,Qwen3-Next 端到端 +3.5%。

## 动机

Qwen3-Next(混合线性注意力架构)在 GDN 路径上大量调用 gated layernorm/rmsnorm(`fla/layernorm_gated`)。旧 Triton 实现 `_layer_norm_fwd_1pass_kernel` 的 grid 为 `(M, ngroups)`,每个 program 只处理一行,长序列(prefill M 上万行)时 program 数巨大、每 program 仅 128 列的工作量,launch 与调度开销占比高,是可优化的热点。PR 正文未给出 profile 细节,直接给出 "long sequence 下 kernel 提速 2x" 的目标,方案改编自 vLLM PR #26207。

## 优化点

核心改动在 Triton kernel 的并行划分:为 kernel 增加 `ROWS_PER_BLOCK` constexpr,program id 映射到行区间起点 `row_start = pid * ROWS_PER_BLOCK`,以 2D tile `[ROWS_PER_BLOCK, BLOCK_N]` 装载 x/z、沿 axis=1 做均值/方差归约、`rstd`/`mean` 变成每行向量后广播,一次性处理多行,grid 缩为 `(cdiv(M, rows_per_block), ngroups)`。`rows_per_block` 按 SM 数自适应:`next_power_of_2(cdiv(M, 2 * sm_count))` 且上限 4,SM 数用 `lru_cache` 缓存;`N` 改为 `tl.constexpr`。同时新增 `cdiv` 工具函数。小 M 时 rows_per_block=1,行为与旧版一致,避免回归。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/layernorm_gated.py: kernel 增加 ROWS_PER_BLOCK 2D tile 化,按 SM 数自适应行数
python/sglang/srt/utils/common.py: 新增 cdiv
benchmark/fla/benchmark_layernorm_gated.py: 新文件,CUDA graph 计时 + 正确性校验
test/srt/test_fla_layernorm_guard.py: 新文件,7 种 case 的精度测试(分布式 spawn)
test/srt/run_suite.py: 将 test_fla_layernorm_guard.py 加入 per-commit 测试套件
```

关键机制(grid 收缩 + 2D tile 归约):

```python
rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
cols = tl.arange(0, BLOCK_N)
x = tl.load(X_base, mask=row_mask & col_mask, other=0.0).to(tl.float32)
var = tl.sum(xbar * xbar, axis=1) / N      # [ROWS_PER_BLOCK]
rstd = tl.rsqrt(var + eps)
...
grid = (cdiv(M, rows_per_block), ngroups)
```

## 性能声明

PR 正文数据:

- kernel 微基准(fp16,x=(65536,128),z gate,rms_norm,CUDA graph 100 次迭代):mean 41.08 us → 17.72 us(约 **2.3x**),吞吐 204.21 → 473.41 GElements/sec;分档:M=4096 3.86→2.29 us,M=16384 11.24→4.48 us,M=65536 41.43→18.13 us,M=131072 80.87→35.08 us,M=262144 159.51→67.63 us;小 M(256/512)持平。自述 "layernorm_gated kernel speedup 2x for long sequence"。
- 端到端(8xH800,Qwen3-Next-80B-A3B-Instruct,TP=4,flashinfer;bench_serving 500 prompts,random input 1024 / output 20,max-concurrency 32):自述 e2e speedup **3.5%**,Mean TTFT 530.22 → 511.09 ms(**3.5%**),Total token throughput 30191.31 → 31006.56 tok/s(**2.6%**)。

## 验证与回归风险

新增 `test/srt/test_fla_layernorm_guard.py`(2 GPU spawn,覆盖 layernorm/rmsnorm/gate 前/后/group norm 共 7 case 及 strided 输入、外部输出 buffer、多维输入),并注册进 per-commit 套件;kernel 微基准内置与 `rms_norm_ref` 的数值对比(max diff < 1e-2 PASS);GSM8K 200 题 accuracy 0.940 自述 "Acc no drop"。风险点:rows_per_block 依赖 SM 数的启发式,异构 GPU 上仅影响分块不改变数值;group norm 路径(groups>1)同样被 2D 化,依赖测试覆盖;NPU 路径(`is_npu`)与 fp32 输入未在 PR 中单独给出数据。

## 关联

同区域(Qwen3-Next / GDN kernel):#15631(CuTe DSL GDN decode)、#19775(fused GDN gating)、#21019(GDN ops 融合)。改编自 vLLM vllm-project/vllm PR #26207。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
