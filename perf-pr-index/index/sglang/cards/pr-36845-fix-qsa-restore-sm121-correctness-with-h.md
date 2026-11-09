# PR #36845: fix(qsa): restore SM121 correctness with Humanize and Kernel Design Agent

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-30 |
| 分类 | Attention / Kernel (major) |
| Label | documentation, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 7 个文件:KDA Triton kernel 包、kernel 注册、后端路由与测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/36845 |

## 一句话结论

为 SM121 新增 Qwen3.8 packed QSA decode 专用 Triton kernel,修复 FlashInfer TRT-LLM 路径静默输出 token 0 的正确性 bug,kernel 层 geomean 2.07x。

## 动机

#36806 的后续,修 #36537 的 SM121/GB10 部分。QSA decode 在 SM120 与 SM121 需要不同路径:精确 SM120 用 FlashInfer TRT-LLM paged decode 数值正确;SM121 上该路径会静默破坏长上下文 decode——120k、190k、210k 复现全部返回 32/32 个 token id 0(`!`),而 server 看似健康、HTTP 200。排除 SM121 可避开错误,但既有 FA4 CuTe varlen 实现在 GB10 上无法编译 QSA 的 packed 单 query shape(`expects coord and shape of view are weakly congruent`)。关闭 FlashInfer autotune 只是让错误 kernel 慢下来,不修数值。

## 优化点

由 Humanize 工作流 + KDA-1.5(Codex 与 Kimi K3 agent)产出、再人工集成的专用 Triton kernel,只针对精确的 Qwen3.8 packed-QSA decode 契约:BF16、D=256、12:1 GQA、TP1 24Q/2KV 或 TP2 12Q/1KV、每序列一条 query、1≤bs≤128、selected KV ≤2055。机制:原正确 Triton 实现每个 query head 一个 CTA,共享同一 KV head 的 12 个 query head 反复重载相同 K/V 行;优化版把这 12 个 head 分组进一个 CTA,用 BF16 tensor-core QK/PV 乘积配 FP32 online-softmax 状态,并把长 KV 行按 split-KV 拆到多个 CTA,最后到达的 split 做 stable FP32 合并并在同一次 launch 里原子计数器清零。split 数只依赖设备端 cu_seqlens 与静态 launch 几何,CUDA Graph 重放无需 host 回读。SM121 直接路由到该 kernel,无 opt-in 开关;不支持的调用抛出描述性契约错误。SM100/SM120 保持 TRT-LLM sparse-decode 路径,其他架构 FA2/FA4 选择不变。

## 关键改动

```text
python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/kernel.py: KDA 生成的 split-KV Triton kernel(BLOCK_M=16、BLOCK_KV 32/64 按拓扑与 batch、MAX_SPLITS=8、scratch 32.3 MiB 上限)
python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/__init__.py: 精确契约 can_use 守卫(dtype/D/GQA/bs/selected-KV/SM121)
python/sglang/kernels/ops/attention/__init__.py: 注册 attention.kda_qwen38_qsa_sm121(SM121-ONLY capability)与 qwen38_qsa_sm121_varlen
python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py: _resolve_flash_attn_varlen_func 在 SM121 优先返回 KDA kernel
test/registered/kernels/test_kda_qsa_sm121.py: 真实 shape、架构门控、CUDA Graph 重放、契约守卫与计数器复位测试
```

split 合并与计数器复位核心:

```python
arrival = tl.atomic_add(counter_ptr + slot, 1, sem="acq_rel", scope="gpu")
if arrival == n_splits - 1:
    ...  # stable FP32 merge of partial max/sum/acc
    tl.atomic_xchg(counter_ptr + slot, 0, sem="release", scope="gpu")
```

## 性能声明

- Kernel 层(对原正确 Triton 实现,隔离测量):15/15 真实 TP1/TP2 张量重放 geomean **2.0702x**(large 1.6951x、small 2.3653x),relative-L2 ≤ 2e-3,每次调用恰好一个 CUDA activity。
- GB10 sweep(TP1/TP2 × bs1-16 × 短/饱和 KV):64/64 通过,max relative L2 0.002422,speedup **1.41x-5.09x**;bs=1 geomean 2.72x、bs=16 geomean 1.83x。扩展 sweep(bs 17/24/32/48/64/96/128):42/42 通过,geomean **4.48x**,最小 1.58x,无回退(bs=128 geomean 10.66x);最坏 TP1 scratch 分配 32.3 MiB。15 万次连续 launch 计数器全部归零。
- E2E(1×GB10/SM121 DGX Spark,CUDA 13.0,`RadixArk/Qwen3.8-Flash-Next-NVFP4` TP1,NVFP4 + NEXTN 3 步,262,144 context,三轮低并发 A/B,45,946 输入 + 1,667 输出 token):concurrency 1 总吞吐 644.51→673.19 tok/s(**+4.45%**)、Mean TTFT 1914.83→1753.04 ms(**−8.45%**);concurrency 4 总吞吐 +4.00%、Mean E2E −2.56%、Mean TTFT −4.20%。
- 准确率:GSM8K 前 50 题 5-shot greedy,两路均 49/50(同一失败例);120k/190k/210k NIAH 精确检索零 token-0 输出。

## 验证与回归风险

`pytest test_qsa.py test_kda_qsa_sm121.py` 在 SM121 上 **73 passed**(删除旧实现后);覆盖真实 shape 参考比对、bs1-128 全覆盖、CUDA Graph 改变设备端 cu_seqlens 重放、150,000 次重复 launch 的原子状态复位、契约守卫(bs 129 拒绝)。CI 状态 Base/Extra/AMD 均红,但 PR 自述这是仓库 rebase gate(要求 main@1ec20fd 而本 PR 有意指向 qwen4-main-squashed),测试 job 被跳过而非测试失败。风险面:kernel 契约极窄,仅 SM121 + 精确 head 拓扑 + bs≤128 + selected KV≤2055 生效,越界直接报错(无静默回退到慢但正确的实现);scratch 按 device 缓存、一次性分配最大 batch;SM120/SM100 路径不变。

## 关联

前序:#36806(本 PR 的 follow-up 目标)、#36537(SM121 正确性问题的原始报告)、#36865(复用的集成结构)。同区域:QSA(Qwen Sparse Attention)backend、FlashInfer TRT-LLM sparse decode、FA4 CuTe varlen(pack 未列编号)。kernel 来源:radixark/KDA-1.5#4(submission b4181149c8884ddb)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
