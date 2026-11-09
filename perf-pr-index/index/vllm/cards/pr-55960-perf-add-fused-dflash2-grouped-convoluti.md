# PR #55960: [Perf] Add fused DFlash2 grouped convolution

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-09-17 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, ci/build, qwen, nvidia, dflash |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/55960 |

## 一句话结论

用一个 Triton custom op 融合 DFlash2 grouped conv，最终调优后 batch256 比已发布融合 kernel 快 1.078×–1.328×。

## 动机

DFlash2 的 grouped dynamic convolution 原先以 Torch 表达式实现。PR 把它融合进单个 Triton custom op，保留原 Torch CPU fallback，目标是在 GPU 上降低该算子的 kernel 开销；实现 docstring 引用 LightSeek TokenSpeed 的上游实现。

## 优化点

在 `qwen3_dflash2.py` 内新增 `_dflash2_grouped_conv_kernel`（一个 Triton custom op，保留 Torch CPU fallback）。采用 row-aligned tiles：至少 128 行且 hidden width 能被 1024 整除时用 1024 元素 tile，否则 512；四 warps、无 autotuner。覆盖非 2 的幂维度、strided coefficient views、多 tap 数、FP32/BF16 两条 tile 路径。并把真实 checkpoint 的 GSM8K accuracy 与 acceptance length 覆盖加入现有 H200 35GB DFlash nightly job，该 case 仍排除在 AMD mirror 之外，Speculators/MTP job 不变。

## 关键改动

```text
vllm/model_executor/models/qwen3_dflash2.py: 新增 Triton 融合 grouped conv kernel 与 custom op 注册
.buildkite/test_areas/spec_decode.yaml: watch 路径加入 qwen3_dflash2.py/qwen3_5.py，AMD mirror 排除 dflash2
tests/v1/spec_decode/test_dflash2.py（新增）: 聚焦 kernel 单元测试
tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py: 真实 checkpoint 精度与验收长度覆盖
```

```python
    row = tl.program_id(0) // triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    col_block = tl.program_id(0) % triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    channels = col_block * ELEMENT_BLOCK + tl.arange(0, ELEMENT_BLOCK)
    mask = channels < NUM_CHANNELS
    groups = channels // GROUP_SIZE
    position = row % BLOCK_SIZE

    delta_row = delta_ptr + row * delta_stride_row
    x_row = x_ptr + row * x_stride_row
    accumulator = (
        tl.load(base_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta_row + groups, mask=mask, other=0.0).to(tl.float32)
    ) * tl.load(x_row + channels, mask=mask, other=0.0).to(tl.float32)
```

## 性能声明

Kernel 微基准（B300、BF16、CUPTI、CUDA graphs、cold L2、四次交替顺序试验取中位；实际 shape hidden 5120、block 8、group 16、two taps、strided 系数；基线为 `torch.compile` 下的原 Torch 表达式）：初始融合 vs Torch，batch 1 1.889/1.920 µs（0.98x）、2 1.02x、4 1.03x、8 1.08x、16 1.21x、32 4.576/3.599 µs（1.27x），均为 kernel-only、batch 1 近似持平。最终 row-aligned 调优 vs 已发布 commit `fd19503785`（11 组 hidden/block、batch 1/32/256）：batch 1 基本不变、batch 32 **1.006–1.172x faster**、batch 256 **1.078–1.328x faster**；代表 batch-256：2560/8 8.384→7.776 µs（1.078x）、5120/8 14.624→11.328 µs（1.291x）、6656/16 35.057→32.049 µs（1.094x）。端到端（B300、TP=1、V2 runner、NVFP4 target `nvidia/Qwen3.8-27B-NVFP4`、BF16 draft `z-lab/Qwen3.8-27B-DFlash2`、全 GSM8K 1319 题 5-shot、temperature 0、max 256 out、七个 speculative tokens）：最终 kernel GSM8K **92.2%**、mean AL **6.14**，与原 Torch 卷积（92.0%/91.2%、AL 6.14/6.13）相当，正文明确这不是 exact-output-equivalence 声明。

## 验证与回归风险

聚焦套件 16 passed（含 1024-tile 对齐与 512-tile 回退）；scratch 边界 sweep 570/570 bitwise comparisons passed（11 组形状、FP32/FP16/BF16、strided/ragged、1/2/3 taps、127/128/129 行边界）；全 NVFP4 模型测试五次各 1 passed（含最终 226.52 s）；changed-file pre-commit 通过（mypy/SPDX/forbidden-import/CUDA API）。风险面：NVFP4 target 把 target-plus-draft model-load memory 从 54.0 GiB 降到 22.85 GiB（非总 serving/KV-cache 内存）；本地验证在 B300，H200 执行与内存适配仍未验证；H200 有 Marlin weight-only fallback；kernel 收益为 kernel-only，非端到端加速；GPU 测试全用 `chg run -g=1 --`。

## 关联

无（PR 正文仅注明 docstring 引用外部 LightSeek TokenSpeed 的 PR 链接，并称 companion 的 TP-reduction / packed-top-k PR 仍独立、未给编号；素材未列可关联的 vLLM PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
