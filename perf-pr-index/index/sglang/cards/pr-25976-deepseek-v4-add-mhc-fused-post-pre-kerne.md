# PR #25976: [DeepSeek-V4] Add mhc_fused_post_pre kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JoeLee314 |
| 合并 | 2026-05-30 |
| 分类 | Attention / Kernel | tier: major |
| Label | deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 展示 5 个文件(`environ.py`、`mhc.py`、`deepseek_v4.py`、`deepseek_v4_nextn.py`、新增 `tests/kernels/test_mhc_kernels.py`),另有 1 个文件从 diff 中略去 |
| 链接 | https://github.com/sgl-project/sglang/pull/25976 |

## 一句话结论

DeepSeek-V4 的 mHC 路径融合 post/pre 边界,小 batch 单次 launch 完成三步工作,DeepSeek-V4-Flash 端到端吞吐 +3.35%。

## 动机

DeepSeek-V4 的 mHC(multi-hyper-connection)路径在延迟敏感的 decode 中串行启动多个 kernel,序列为 `mhc_post -> pre-norm GEMM -> mhc_pre finalization`。对小 token batch,每次边界都要付出 launch 开销与中间张量往返。方法改编自 TRTLLM:保留数值敏感的 `mhc_pre_big_fuse(_with_norm)` finalization 阶段原样不动,只消除 post/pre 之间的独立调度边界。

## 优化点

新增 `mhc_fused_post_pre` 分发层与两条路径。小 batch(≤32 token)走新 TileLang scalar-FMA kernel `mhc_fused_post_pre_fma_tilelang`,一次 launch 完成:上一跳 hc_post(post/comb 系数作用于 hidden 与 prev_residual)、bf16 residual 物化(与 unfused 的中间精度一致:mhc_post 写 bf16、mhc_pre 读 bf16)、pre-norm GEMM 部分和、RMS square-sum 部分和;CTA 按 token x mix-output-tile x hidden split-k 网格展开,warp 内 reduce 后写 split partials,交由不变的 `mhc_pre_big_fuse(_with_norm)` 汇总并做 sinkhorn/finalize。大 batch 保留现有高吞吐 TileLang `mhc_post` + DeepGEMM 分解(tensor core 路径不被替换),只融合 Python/model 层调度边界。模型层配合:跨层融合(前层 FFN `hc_post` 与下一层 attention `hc_pre` 合并)与层内融合(attention `hc_post` + FFN `hc_pre`);最后一层 FFN 的 `hc_post` 延迟到 model tail 统一闭合;NextN 解码器适配 deferred 状态;缓存 bf16 RMSNorm 权重避免每次 forward cast;提供 fused 路径 prewarm。零 token 的 DP/EP rank 返回类型正确的空 tensor。整条路径由 `SGLANG_OPT_FUSE_MHC_POST_PRE=1` 门控(默认关),并显式依赖既有 `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST` 开关——fused 实现直接复用 TileLang mHC 的 tensor layout 与 finalization kernel。

## 关键改动

```text
python/sglang/srt/layers/mhc.py: 新增 mhc_fused_post_pre_fma_tilelang 与 mhc_fused_post_pre 分发(小 batch FMA / 大 batch DeepGEMM)
python/sglang/srt/models/deepseek_v4.py: 跨层/层内 mHC 融合调用、deferred hc_post、bf16 norm 权重缓存、prewarm
python/sglang/srt/models/deepseek_v4_nextn.py: NextN 单层 decoder 收尾 deferred hc_post
python/sglang/srt/environ.py: SGLANG_OPT_FUSE_MHC_POST_PRE 开关
tests/kernels/test_mhc_kernels.py: 新增 fused vs unfused 一致性测试
```

```python
# The scalar-FMA kernel wins only for small batches where launch
# overhead dominates; beyond the threshold DeepGEMM's tensor-core path wins.
fma_token_threshold = 32
if num_tokens <= fma_token_threshold:
    ...
    mhc_fused_post_pre_fma_tilelang(...)
else:
    mhc_post_tilelang(...)  # + DeepGEMM prenorm GEMM
```

## 性能声明

- `bench_serving` on DeepSeek-V4-Flash:total_throughput increased by **3.35%**(Performance summary 为截图,正文未给出吞吐绝对值、硬件与负载参数)。
- 准确性:GSM8K score 0.975(GSM8K-200 端到端)。
- 其余场景(不同 batch/硬件)的数字:PR 未注明。

## 验证与回归风险

测试:新增 `tests/kernels/test_mhc_kernels.py`,按 hidden_size {4096, 7168} x num_tokens {0, 1, 8, 17, 32, 64} x use_norm {False, True} 参数化,断言 `mhc_fused_post_pre` 与 unfused 参考序列一致(需 CUDA,TileLang mHC kernel);作者 checklist 声明已验证 operator-level precision、short deterministic sanity prompts、GSM8K-200 端到端、bench_serving 性能。风险:默认关闭(env 门控),依赖 TileLang mHC pre/post 开关同时开启;FMA kernel 仅覆盖 bf16 residual/fp32 mix 的 dtype 组合(有 assert);32 token 阈值与 tile/split 参数为静态启发式;deferred hc_post 改变层间数据流,cuda graph/piecewise 路径需 prewarm 支持;CI 状态显示 Base/Extra 均失败(合并时的外部状态,正文未说明)。

## 关联

TRTLLM(方法改编来源);复用同文件内的 `mhc_pre_big_fuse(_with_norm)`、`mhc_post_tilelang`、DeepGEMM `tf32_hc_prenorm_gemm`(同区域 mHC kernel 栈)。无正文列出的其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
