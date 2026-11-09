# PR #40640: [Kimi K3] Fix CUDA graph stream explosion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kpham-sgl |
| 合并 | 2026-09-22 |
| 分类 | Speculative Decoding |
| Label | (无) |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/40640 |

## 一句话结论

修复 Kimi K3 target CUDA graph replay 流膨胀:96 流降到 4,TP8 BS1 吞吐约 +1%。

## 动机

Kimi K3 的 target CUDA graph replay 在 B300 上膨胀到 96 个流:重复的 attention fork 把后续层挪到新的 replay 流上。需要在辅助 kernel 之前先记录主 attention 分支,抑制流爆炸。

## 优化点

两处 capture 顺序重构,均只影响 capture mode:
1. KDA `forward_qkvbfg_fused`:原来在 side stream 先发 [f_a|b]+f_b 小 GEMV、主 stream 后发宽 [q,k,v,g] GEMM;现改为 fork 两个分支后先在主 stream 记录 `fused_qkvg_proj` 主投影,再在 alt stream 记录 gate GEMV,避免 replay 时逐层分裂出新流。
2. MLA output gate:`_precompute_output_gate`(fork 即在 alt stream 发 gate GEMM)改为 `_fork_output_gate`(只 fork 流、记下 pending stream),gate kernel 延到 attention core 之后、o_proj 边界由 `_compute_output_gate` 记录,fork/join 保持在同一 capture segment 内;即使 gate 分支被跳过,也补一次 `wait_stream` 关闭捕获分支,消除跨段依赖。

## 关键改动

```text
python/sglang/srt/models/kimi_k3.py: KDA 主投影提前记录;MLA gate fork 与 record 分离;skipped gate 分支闭合
test/registered/unit/models/test_kimi_k3_bfa_overlap.py: 改为 changed-input replay parity,新增 MLA gate capture 测试
```

## 性能声明

8× B300、CUDA 13.0、TP8、60k input / 1k output、dummy MXFP4 target/draft 权重、模拟 acceptance 5.5、FP8 E4M3 KV、FP32 SSM、CuTe DSL MLA、FlashInfer MXFP4 MoE,N=5(每变体每 batch size 5 次未 profile 运行):

- TP8/DCP1:BS1 493.06→498.12 tok/s(+1.03%,5 对全部提升 +0.77%~+1.14%);BS16 3139.65→3249.41 tok/s(+3.50%),但均值被一个慢 baseline 样本(2689.09 tok/s)扭曲,其余 4 对为 −0.10%~−0.15%,作者声明不构成 BS16 加速结论。
- TP8/DCP8(fi_a2a、replicated Q projection):BS1 470.94→474.87 tok/s(+0.83%,全部提升);BS16 3060.60→3060.26(−0.01%,不变)。
- Profile:8 个 TP rank 的 target replay 流均从 96 降到 4,16 次 profiled replay 分配稳定,kernel name/count multiset 不变(每 replay 1,950 个 target kernels)。

## 验证与回归风险

`test_kimi_k3_bfa_overlap.py` 5 个测试在 B300 通过:KDA bitwise parity 覆盖 token 数 1/4/12、两种 `defer_f_b` 设置、3 次 changed-input replay;MLA gate 覆盖 in-graph input producer、changed-input parity 与 eager/breakable-graph fallback 选择;Overlap unit CI 与 B300 e2e CI 通过。风险:fork/join 顺序变化若与未来 breakable-graph 段切分交互可能引入隐藏依赖;N=5 样本量小,dummy 权重与模拟 acceptance 不检验模型精度;仅 B300 实测。

## 关联

正文未列同区域其他 PR 编号。无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
