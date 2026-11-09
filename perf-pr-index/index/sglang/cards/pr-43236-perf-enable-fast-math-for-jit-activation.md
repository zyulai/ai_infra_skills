# PR #43236: [Perf] Enable fast-math for JIT activations on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(素材未提供统计);diff 为单文件 8 行级改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/43236 |

## 一句话结论

Blackwell 上 JIT 激活 kernel 恢复 fast-math 编译,SiLU-and-mul 每层 kernel 时间降最多 1.55x。

## 动机

Blackwell 上 JIT 激活 kernel 因历史政策未开 fast-math:SiLU 的 `x / (1 + expf(-x))` 编译成 IEEE 除法,在 sm_103a SASS 中每个元素都有自己的 `BSSY`/`BSYNC` 区域(`MUFU.EX2` → `MUFU.RCP` → Newton `FFMA` → `FCHK` → 条件 `CALL` 慢路径),单线程 16 次 `FCHK`/`CALL`,元素串行而非交错,约 46 ns/元素;decode 规模下 warp 太少无法隐藏,SiLU-and-mul 单 token 也要 1.7–1.8 µs/层。这一 precise 政策源自 AOT sgl-kernel 为 SM100 FP8 blockwise GEMM 引入的例外(#4278、#4376),#10701 又把 AOT `common_ops` 拆成 SM90 fast-math / SM100+ precise 两套构建,JIT 激活(#22094)照搬了该政策;而 Hopper 一直跑 fast-math。

## 优化点

`get_activation_cuda_cflags()` 改为只要 `is_cuda()` 就返回 `["--use_fast_math"]`,删除 SM100+ 豁免分支。fast-math 把 IEEE 除法变成无分支近似除法、启用近似 `expf`、并把 subnormals flush 到零。受影响 kernel:门控激活 `silu_and_mul`/`gelu_and_mul`/`gelu_tanh_and_mul`、默认 `fast_math=True` 的 `relu2`(需要 BF16 subnormal 的调用方如 Cosmos3 已传 `fast_math=False`)、`situ_and_mul`。带 rounding 的变体(`*_with_activation_rounding`)仍构建 precise 的 `rounded_activation` 模块;HIP 保持无 fast-math。

## 关键改动

```text
python/sglang/kernels/jit/utils/arch.py: 唯一改动文件,激活 kernel 的 CUDA cflags 策略
```

```python
def get_activation_cuda_cflags() -> list[str]:
    # 旧逻辑:HIP 或 SM100+ 返回 [](precise),其余 fast-math
    if is_cuda():
        return ["--use_fast_math"]
    return []
```

## 性能声明

B300(sm_103,148 SMs)BF16 SiLU-and-mul,仅激活:36 个不重叠 layer buffer 捕进一条 CUDA graph,20 次 replay、11 次取中位数,main 与本 PR 同卡交替各 3 轮:

| hidden × tokens | main µs/层 | 本 PR µs/层 | 加速 |
| ---: | ---: | ---: | ---: |
| 2432 × 8 | 1.77 | 1.14 | 1.55x |
| 3584 × 32 | 1.94 | 1.31 | 1.48x |
| 9728 × 8 | 1.94 | 1.31 | 1.48x |
| 9728 × 128 | 3.52 | 2.64 | 1.33x |
| 13824 × 512 | 9.01 | 7.62 | 1.18x |
| 13824 × 2048 | 28.73 | 25.97 | 1.11x |

28 个 shape(hidden 2432/3584/9728/13824 × tokens 1–2048)总时长 134.6 → 108.8 µs,每个 shape 至少 1.11x。PR 明确说明这是 kernel 计时而非服务吞吐;仅测了 B300,Hopper 本就 fast-math、无变化。

## 验证与回归风险

数值影响:Blackwell 输出与 main 不再 bitwise 一致,差异几乎全部是 BF16 subnormal 被 flush 为 0。SiLU-and-mul 在 B300 上穷举全部 65,536 个 gate bit pattern:BF16 有 0.78–1.22% 输出不同(均为 |g|<1e-36 或 g<-80 的 FTZ),FP16 仅 1–3 个 1-ulp 舍入翻转;对 fp64 参考的 correctly-rounded 比例从 65,263/65,280 降到 64,752。端到端:`test_inkling_small_nvfp4.py`(Inkling-Small-NVFP4,TP4,DSpark 投机解码,确定性推理)fast-math 与 precise 均通过 3 项 KL 检查。单元测试共约 540 项通过(activation 452 passed + situ_and_mul、diffusion、namespace 等)。回归风险:MUSA 路径现在跟随 `is_cuda()` 拿到 fast-math,作者声明未测试;BF16 极小值语义变化对依赖 subnormal 的调用方可见(此类调用方需显式走 precise 变体)。

## 关联

前序:#4278、#4376(SM100 AOT 弃用 fast-math 的起点)、#10701(AOT common_ops 拆分)、#22094(JIT 激活引入)、#41887(引入确定性回归,被本 PR 的 CI 暴露)、#43447(修复该回归)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
