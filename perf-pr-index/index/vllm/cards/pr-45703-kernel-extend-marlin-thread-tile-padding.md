# PR #45703: [Kernel] Extend Marlin thread-tile padding to MoE (WNA16 + FP8/MXFP8)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-06-23 |
| 分类 | MoE(tier: major) |
| Label | bug, performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/45703 |

## 一句话结论

Marlin MoE 在 TP 分片后 intermediate 不对齐 thread tile 时也能走 Marlin,避免回退慢速 kernel 或 repack 崩溃。

## 动机

前作 #45295 已给 dense Marlin 与 NVFP4 MoE 加了 thread-tile padding,但其余 Marlin MoE prep 未覆盖。TP sharding 会使 rank-local intermediate size 落不到 Marlin thread tile 上:此时 WNA16 MoE(GPTQ/AWQ/compressed-tensors)直接拒绝 Marlin、回退到更慢的 kernel,FP8/MXFP8 MoE 则在 repack 阶段崩溃。本 PR 把同一 padding 机制扩展到这些剩余路径。

## 优化点

权重准备阶段把 intermediate 零填充到最小合法 tile(kernel 要求 gate-up 2*N%128、down N%64,并保持 group_size 整除)。padding 区域不会进入 MoE 输出:WNA16 靠 zero-padded scales 抵消(w13 的 padded 输出通道为 0,故 w2 的 padded 输入为 0);FP8/MXFP8 的量化零解码为 0.0,padded 权重直接无贡献。apply 路径从 packed weight shape 反推 padded N,不需要透传新 kwargs。hidden_size(MoE I/O 维度)永不 pad;act-order 与跨界 group 保持严格检查;Block-FP8(weight_block_size=[128,128])本就要求 intermediate%128,padding 为 no-op。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/marlin_utils.py: 新增 marlin_moe_padded_intermediate() 与 check_moe_marlin_supports_layer(allow_tile_padding=)
vllm/model_executor/layers/fused_moe/oracle/int_wna16.py: _pad_w13_shard_cols/_pad_rows/_pad_w13_bias 零填充 helper,接入 GPTQ/AWQ 两条 prep
vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py: FP8/MXFP8 MoE prep 中 pad 权重与 scale
vllm/model_executor/layers/quantization/{auto_awq,auto_gptq}.py 等: MoE gate 打开 allow_tile_padding(act-order 除外)
tests/kernels/quantization/test_marlin_tile_padding.py: CPU 策略/门测试 + GPU round-trip
```

```python
def marlin_moe_padded_intermediate(intermediate_size: int, group_size: int = -1) -> int:
    group = group_size if group_size > 0 else 1
    padded = round_up(intermediate_size, math.lcm(64, group))
    if padded != intermediate_size:
        logger.warning_once(
            "Marlin requires thread-tile padding for the MoE intermediate size ...")
    return padded
```

## 性能声明

PR 正文未给出 benchmark 数字(regex 亦未检出)。机制收益为:原本回退 WNA16 慢速 kernel 或 FP8/MXFP8 repack 崩溃的 shape 现在可走 Marlin。PR 同时如实注明:命中 padding 时每次 forward 会对 padded experts 做 pad/slice,"performance may be degraded"。

## 验证与回归风险

tests/kernels/quantization/test_marlin_tile_padding.py:95 passed, 13 skipped。新增 CPU policy/gate 测试与 GPU round-trip(真实 prepare + repack + fused_marlin_moe vs 反量化参考,在 tile 不对齐的 intermediate 上):WNA16 GPTQ int4(channelwise+groupwise)精确匹配(对称 int4 零解码为 -8,检验 zero-padded scales 抵消);FP8/MXFP8 在 fp8 精度内匹配。风险:padding 后 forward 有 pad/slice 开销;act-order=group 不支持 padding;hidden_size 不 pad,仍按严格检查拒绝。

## 关联

前序:#45295(dense Marlin 与 NVFP4 MoE 的 thread-tile padding,本 PR 为其明确 follow-up)。同分类 MoE 卡:#19205、#39904(tuning configs)、#33506、#46518(FlashInfer MoE 后端能力)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
