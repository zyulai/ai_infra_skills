# PR #47327: [1/N] Add dense MHA path for sparse MLA short sequences

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-07-14 |
| 分类 | Attention / Kernel |
| Label | documentation, performance, rocm, intel-gpu, ready, v1, nvidia |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面);diff 因超长截断,另有 16 个文件未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/47327 |

## 一句话结论

batch 内全部 prefill 的 seq_len<=topk 时改走 dense MHA 而非 sparse MQA,GLM-5.2 无共享前缀负载总吞吐 +24.5%。

## 动机

sparse MLA(DSA)对每个 query token 只做 top-k(2048)token 的 MQA gather。当序列本身就短于 topk 时(indexer 会选中几乎全部 KV),稀疏 gather 的索引间接寻址与内核开销反而不如直接做 dense attention;但此前 sparse MLA 后端一律走 forward_mqa,短序列 prefill 付出不必要的代价。

## 优化点

为 sparse MLA 后端增加条件快速路径:当 batch 内所有 prefill 的 `seq_len <= topk`(且满足 q_len 阈值条件)时路由到 dense MHA(`forward_mha`,`sparse_mla_mha_mode="dense"`),否则维持 sparse MQA;机制上通过 `sparse_mla_force_mqa` 配置与 `reorder_batch_threshold` 控制路径选择(基准中用 -1 阈值强制短 extend 也走 prefill/MHA 路径)。本 PR 同时搭建了 `mha_vs_mqa` 层级基准模式(新增 `configs/mla_sparse_mha_vs_mqa.yaml`,覆盖 BS 1-32 x seq_len 32-2048 网格、random/prefix/sliding_window 三种 topk 指纹)并把 kv_b_proj mock 换成真实 `ColumnParallelLinear`。这是 masked MHA(#34744,支持 seq_len>topk)的前置 [1/N]。

## 关键改动

```text
vllm/v1/attention/backends/mla/(sparse 后端,文件在截断部分): seq_len<=topk 时 dense MHA 路由
benchmarks/attention_benchmarks/{benchmark,common,mla_runner}.py: mha_vs_mqa 模式 + sparse_mla_* 配置
benchmarks/attention_benchmarks/configs/mla_sparse_mha_vs_mqa.yaml: 新增对比基准配置
```

## 性能声明

e2e(DSV3.2,TP8+EP,1024-2048 ISL / 500 OSL,2000 prompts):base total 12,544 tok/s / output 3,087 tok/s / median TPOT 179.7 ms → PR 13,399 / 3,297 / 175.8 ms,+6.8%。GLM 5.2(TP8+EP,1500 ISL / 500 OSL):无共享前缀 main 19,452 → PR 24,224 tok/s(+24.5%),median TTFT 2.10→1.46 s(-30%),mean TPOT 95.9→76.6 ms(-20%);有 1000-token 共享前缀 29,900 → 30,464(+1.9%),TPOT +1.7%。层级微基准(B200 DSv3.2/GLM5.x、H100,FLASHMLA_SPARSE/FLASHINFER_MLA_SPARSE/FLASH_ATTN_MLA_SPARSE decode 后端)以图形式给出,正文无数值。精度:gsm8k flexible 0.9462→0.9439,strict 0.9447→0.9439。

## 验证与回归风险

正确性由 CI 覆盖("Verify correctness with CI");层级基准矩阵覆盖 sparse MLA 各后端与 topk 指纹。风险:dense MHA 路径仅对 seq_len<=topk 生效,路由条件错误会影响稀疏语义(依赖全部 prefill 满足条件的批量级判定);H100 GLM5.x 维度不支持;共享前缀(prefix caching)场景收益收窄至 +1.9% 且 TPOT 略负。

## 关联

系列:[1/N],后续 #34744(masked MHA,seq_len>topk 加速)。同区域(sparse MLA/DSA):PR #37421、PR #46076、PR #49792。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
