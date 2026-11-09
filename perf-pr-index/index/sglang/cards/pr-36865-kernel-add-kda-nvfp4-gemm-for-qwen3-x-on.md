# PR #36865: [Kernel] Add KDA NVFP4 GEMM for Qwen3.x on SM120

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-02 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, quant, blackwell, run-ci, jit-kernel |
| Diff | pack diffstat 缺失;diff 含新 kda_kernels 包(CuTe DSL GEMM 约 2500 行)及多个 diffusion kernel 文件迁入(pack 另注明 9 个文件省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/36865 |

## 一句话结论

接入 Kernel Design Agents 生成的 Qwen3.x NVFP4 SM120 GEMM 并按 shape allowlist 派发:16 行生产 shape 几何平均 1.319x,Qwen3.5-4B e2e 吞吐最高 +8.73%。

## 动机

为 Kernel Design Agents / Humanize2 工作流生成的 kernel 建立落地机制,并把首个 E2E 合格的成果——Qwen3.x ModelOpt NVFP4 SM120 GEMM(来自 BBuf/KDA-Pilot#195 @ `516c976c`)——接入生产派发。取代已关闭的 #36043。关键教训:原隔离 GEMM 策略会把每层 5.6–11 MiB 的 weight-scale 张量持久驻留 L2,单 kernel benchmark 受益,但在完整 serving pipeline 中挤掉 attention/SSM state——启用全部已捕获 Qwen3.8 decode shape 会让端到端输出吞吐回退 0.76%,因此派发契约必须按 E2E 验证过的 shape 白名单收敛。

## 优化点

新增 `python/sglang/kernels/kda_kernels` 作为 agent 生成 kernel 的实现家(provenance 记为 `KernelBackend.KDA`,与实现语言无关),现有 KDA diffusion 实现与其 JIT CUDA sources 迁入,`sglang.kernels.ops` 保持稳定 public facade、拥有注册与 fallback 策略。GEMM 本体是从 CUTLASS dense block-scaled GEMM example 特化的 SM120 warp-MMA kernel:m16n8k64 `MmaMXF4NVF4Op` atoms、TMA producer warp、无 TMEM/tcgen05/2-CTA,tile 固定 16x64x512、cluster (1,1,1),FP4-only、FP32 累加,使用 L2 `evict_first`/`evict_last` cache policy 内联汇编。派发侧 `try_qwen3x_nvfp4_gemm` 精确校验 dtype(uint8 packed FP4)、layout、scale factor 形状、SM120 capability 与 shape allowlist,任何不满足的组合回退 FlashInfer。Qwen3.8 只放行 DSpark 验证过的 down projection `(M,K,N)=(9,17408,5120)`,并把 weight 与 scale 改为 one-pass streaming。

## 关键改动

```text
python/sglang/kernels/kda_kernels/: 新包,agent 生成 kernel 的实现家(README 记录各 kernel provenance)
python/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py: CuTe DSL SM120 NVFP4 GEMM 实现
python/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py: 惰性派发门(shape allowlist + dtype/layout/capability 校验)
sglang.kernels.ops 下多个 diffusion kernel 文件: 迁入 kda_kernels,公共 facade 不变
```

## 性能声明

硬件:RTX PRO 6000 Blackwell Server Edition(SM120, 96 GB),PyTorch 2.13.0+cu130,flashinfer-python 0.6.18;模型 AxionML/Qwen3.5-4B-NVFP4 与 Qwen3.5-9B-NVFP4。

- Kernel 级(multi-weight benchmark:8 个不同 weight 在 CUDA Graph 内轮换,FlashInfer/KDA 交替计时 5 trials,取中位):20 行全部通过 `rtol=1e-2, atol=2e-2`。Qwen3.5-4B down 2.02–2.05x、gate/up 1.03–1.06x;Qwen3.5-9B gate/up 1.12–1.14x、down 1.25–1.26x;几何平均 **1.319x**(16 个生产派发的 M=1/2/4/8 行)、1.318x(全部 20 行)。
- E2E 并发 sweep(32 条固定 seed random-ids 请求,2048 输入/512 输出 token,conc 1/2/4/8,每轮前 flush cache,3 轮 baseline + 3 轮 KDA + 1 轮相邻 baseline):Qwen3.5-4B 吞吐 +8.73% / +6.84% / +7.12% / +6.52%(conc 1/2/4/8),TPOT 与 E2E latency 同步改善;Qwen3.5-9B +3.18% / +2.78% / +2.82% / +2.70%。相邻 baseline 与原 baseline 均值差 <0.32%;TTFT −1.93%~+3.57%。
- Qwen3.8-27B DSpark 三轮 mean:128.35 → 129.60 output tok/s(+0.98%),accept length 与 fixed-prompt 输出不变(此前全 shape 放行会 −0.76%)。

## 验证与回归风险

GPU correctness 对照 FlashInfer:Qwen3.5 M=1/2/4/8/9 decode shapes、Qwen3.8 decode/prefill、CUDA Graph replay、ModelOpt dispatch、unsupported-shape fallback。Fused-op dispatch tests 56/56;LayerNorm fused-op backend 20 passed + 1 环境 skip;kernel namespace tests 61/61;pre-commit 全过。风险:shape allowlist 极窄,依赖 cutlass 与 `cuda.bindings.driver` 运行时,不满足即整体回退 FlashInfer;Qwen3.8 路径的窄化决策来自 L2 驻留的负面教训,新增 shape 需重做 E2E 验证。

## 关联

#36043(被本 PR 取代并关闭);#34934(benchmark 协议);上游来源 BBuf/KDA-Pilot#195。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
