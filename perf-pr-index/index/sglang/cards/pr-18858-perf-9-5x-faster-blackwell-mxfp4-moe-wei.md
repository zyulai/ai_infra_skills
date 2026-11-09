# PR #18858: [Perf] ~9.5x faster Blackwell MXFP4 MoE weight loading

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-02-16 |
| 分类 | MoE (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 可见部分仅 mxfp4.py 一个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/18858 |

## 一句话结论

gpt-oss MXFP4 MoE 加载时的逐 expert shuffle 改为缓存 permute-index 重排,Load weight 264.78s 降到 28.03s(9.45x)。

## 动机

Blackwell GPU 上 gpt-oss MXFP4 MoE 的加载期预处理过慢:每个 expert 的 weight/scale/bias 都要调用 `shuffle_matrix_a`/`shuffle_matrix_sf_a` 做 shuffle,36 层逐层日志显示每层约 6 秒,总 Load weight elapsed 264.78 秒,其中绝大部分是 shuffle 预处理。目标是把加载期权重布局预处理的开销降下来。

## 优化点

机制改变:同一形状张量的 shuffle 是一个固定置换,因此把"逐 expert 调 shuffle kernel"改为"每张量形状算一次 permute index,之后所有 expert 直接用索引 gather"。具体用 FlashInfer 的 `get_w2_permute_indices_with_cache`,host 侧缓存按 `torch.Size` 键;再叠加一层 device-resident 索引缓存(键为 shape/epilogue_tile_m/num_elts_per_sf/device),避免把索引反复 H2D 传输。per-expert 循环退化为 `weight_uint8[permute_indices].contiguous()`。scale 张量按 MXFP4 kernel 要求的布局,在 permute 后经 `nvfp4_block_scale_interleave` 输出 kernel-ready interleave 格式(weight/bias 用默认模式,scale 用 `num_elts_per_sf=16`);`epilogue_tile_m=128` 暂为硬编码(代码注释 FIXME)。同时删除逐层 shuffle 日志。这是纯加载期布局预处理改动,数学/前向路径预期不变。

## 关键改动

```text
python/sglang/srt/layers/quantization/mxfp4.py: 新增两级 permute-index 缓存并重写 process_weights_after_loading 的 shuffle 逻辑
```

```python
def _get_flashinfer_mxfp4_device_permute_indices(
    x, epilogue_tile_m, num_elts_per_sf=None,
) -> torch.Tensor:
    permute_indices = get_w2_permute_indices_with_cache(
        _flashinfer_mxfp4_permute_indices_cache, x, epilogue_tile_m, **extra_args,
    )
    cached_device_indices = _flashinfer_mxfp4_permute_indices_device_cache.get(cache_key)
    if cached_device_indices is None:
        cached_device_indices = permute_indices.to(x.device)
        _flashinfer_mxfp4_permute_indices_device_cache[cache_key] = cached_device_indices
    return cached_device_indices
```

```python
gemm1_weights_mxfp4_shuffled.append(
    w13_weight[i].view(torch.uint8)[w13_weight_permute_indices].contiguous()
)
```

## 性能声明

测试条件:GptOssForCausalLM,dtype torch.bfloat16,Blackwell GPU(pack 未注明具体型号;before 日志 avail mem=197.20 GB,mem usage=69.69 GB)。指标为 `Load weight end` elapsed:

- Before:264.78 s
- After:28.03 s
- Delta:−236.75 s
- Speedup:9.45x(减少 89.4%)

PR 未提供推理吞吐(而非加载时间)的对比数据。

## 验证与回归风险

无数学/前向行为改变的预期,PR 用 GPQA 验证:`gpt_oss.evals --model openai/gpt-oss-120b --eval gpqa --reasoning-effort low` 得分 0.6528;pack 中未给出 BF16 基线对照数字。Checklist 各项(格式/单测/文档/benchmark)在 PR 快照中均未勾选。回归面:仅 FlashInfer MXFP4 路径的 `process_weights_after_loading`;依赖 flashinfer 提供 `get_w2_permute_indices_with_cache` 与 `nvfp4_block_scale_interleave`;两级缓存以 shape/device 为键,若 kernel 的 `epilogue_tile_m` 内部假设变化,硬编码 128 需同步更新。

## 关联

同区域:gpt-oss MXFP4 MoE 在 Blackwell 上的 FlashInfer `trtllm_fp4_block_scale_moe` 执行路径(本 PR 仅改其加载期预处理);pack 正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
