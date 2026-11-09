# PR #39130: Add triton autotune on the Mamba2 SSD kernels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @elvischenv |
| 合并 | 2026-10-01 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, jit-kernel |
| Diff | diffstat 缺失(素材未提供增删行统计,diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/39130 |

## 一句话结论

为 Mamba2 SSD Triton 内核恢复 `@triton.autotune`,修复所有内核静默运行在 `BLOCK_SIZE_*=16` 默认值的 10 倍级损失,端到端提升至 1.28x。

## 动机

SGLang 的 Mamba2 SSD Triton 内核自 vLLM 移植时未携带 `@triton.autotune` 装饰器:最初导入(#10988)把装饰器注释掉了,后续 #10909 在重新同步 ops 时删除了这些注释块。而 launcher 仍按 `META["BLOCK_SIZE_*"]` 计算 grid,于是每个内核一直在签名默认 `BLOCK_SIZE_* = 16` 下静默运行——按 PR 说法在 prefill 内核上是 10x+ 的损失。上游 vLLM 至今这些装饰器都是生效的。

## 优化点

为 `ssd_bmm`、`ssd_chunk_scan`、`ssd_chunk_state`、`ssd_state_passing` 恢复 `@triton.autotune`,config 列表与 key 逐字取自当前 vLLM 对应目录,保证下次上游同步仍可 diff。sglang 独有的 `_chunk_state_varlen_kernel` 复用 `_chunk_state_fwd_kernel` 的配置表(同 tile-shape 家族)。接入既有 `**autotune_cache_kwargs`(FLA 惯例),使调参结果跨进程重启持久化而非每次启动重新测。新增 `prune_oversized_tiles`:把每个 `BLOCK_SIZE_*` 上限截到其所 tile 维度的 `next_power_of_2`,因为超过维度的 tile 只会在相同 grid 上做全 mask 计算、必然被支配——这也是修复 `test_mamba_ssm_ssd.py` CI 30 分钟超时的手段。另修复 `bench_one_batch` 在 Mamba 模型上 decode CUDA-graph capture 崩溃(进程级 backend 初始化提前到 `load_model` 之前)。

## 关键改动

```text
python/sglang/kernels/ops/mamba/triton_ops/autotune.py: 新增 prune_oversized_tiles 与 autotune_cache_kwargs 转出
python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_scan.py: _chunk_scan_fwd_kernel 挂 23 项 autotune 配置表
python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_state.py: _chunk_state_fwd/_chunk_state_varlen 挂 autotune
python/sglang/kernels/ops/mamba/triton_ops/ssd_bmm.py / ssd_state_passing.py: 挂 autotune
python/sglang/benchmark/one_batch.py: _init_process_global_configs 供 correctness_test 复用
```

关键片段(超大 tile 剪枝):

```python
def early_config_prune(configs, named_args, **kwargs):
    args = {**kwargs, **named_args}
    caps = {}
    for block, dims in tiled_dims.items():
        cap = triton.next_power_of_2(min(args[dim] for dim in dims))
        caps[block] = max(cap, min(c.kwargs[block] for c in configs))
    keep = [c for c in configs if all(c.kwargs[b] <= cap for b, cap in caps.items())]
    return keep or [min(configs, key=_tile_footprint)]
```

## 性能声明

端到端 serving:Nemotron-3-Ultra-550B-A55B-NVFP4,GB300 TP4,ISL 8192 / OSL 1024。Total tok/s(main → +autotune):concurrency 16:10,981→13,264(1.21x);32:15,728→17,751(1.13x);64:20,772→23,662(1.14x);128:24,877→29,892(1.20x);256:28,190→35,747(1.27x);512:29,558→37,776(1.28x)。tok/s/user 提升介于 1.05x-1.22x。准确性:lm_eval GSM8K(5-shot)同模型 TP4,flexible-extract 0.9401→0.9409、strict-match 0.9378→0.9393(误差范围内持平)。PR 未给出内核级微基准数字。

## 验证与回归风险

PR 自述 CI 成本:冷启动 pytest 下 `_chunk_scan_fwd_kernel` 单核约 22 个 key × 23 配置 ≈ 500 次 compile+bench,六内核合计约 4 倍,曾造成 30 分钟超时;prune 将小 CI 形状收敛到每 key 1 个配置而不改测试矩阵。注意 autotune 选择 `BLOCK_SIZE_K` 会改变 fp32 `tl.dot` 累加的 K-split,输出与旧默认值不 bit-identical,以 GSM8K 分数持平佐证。风险路径:所有 Mamba2/混合线性模型(如 Nemotron、Qwen3-Next 类)的 prefill/decode SSD 内核;首次命中新 key 时的调参开销;持久缓存目录的行为。

## 关联

前序:#10988(SSD 内核引入,装饰器被注释)、#10909(上游同步删除注释块)。同分类:vLLM 上游 Mamba ops 同步、FLA autotune cache(`kernels/ops/attention/fla/utils.py`)相关改动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
