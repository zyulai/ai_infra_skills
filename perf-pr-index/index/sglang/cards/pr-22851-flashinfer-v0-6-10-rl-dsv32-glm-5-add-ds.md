# PR #22851: [FlashInfer v0.6.10] [RL] [DSv32] [GLM-5] Add `--dsa-topk-backend` and integrate FlashInfer and pytorch topk

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zianglih |
| 合并 | 2026-05-25 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);3 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/22851 |

## 一句话结论

为 DSA indexer top-k 增加可配置后端选择(`--dsa-topk-backend`:sgl-kernel/torch/flashinfer),接入 FlashInfer 可确定性、可配 tie-break 的 topk 以满足 GLM-5 RL 需求。

## 动机

GLM-5 的 RL 流程使用 `torch.topk`;而 FlashInfer topk(flashinfer#3095)提供确定性执行与可配置 tie-break,并且按 PR 正文说法在长上下文下性能更好。此前 DSA indexer 的 topk 硬编码 sgl-kernel 实现,无法按模型/场景切换。

## 优化点

(1) 新增 `--dsa-topk-backend` 服务参数(默认 `sgl-kernel`,可选 `torch`/`flashinfer`)与 `dsa_topk_backend.py`:统一 `topk_func`(非融合路径)与 `topk_transform`(融合路径,PAGED/RAGGED 两种变换)的 backend 分发;`DSAIndexerMetadata` 增加 `topk_backend` 字段,原 dsa_backend 内联的 sgl-kernel 调用全部改走该抽象。非融合路径统一 `_topk_unfused`:按 row 窗口 mask(-inf 填充)后调用注入的 topk 算子(torch.topk 或 flashinfer.top_k),越界槽位填 -1。(2) FlashInfer 侧暴露两个环境变量:`SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC`(确定性 kernel)与 `SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK`(`small`/`large` 决定同分时候选索引取小/取大;设置后强制确定性 topk);fused 路径接入 `flashinfer.top_k_page_table_transform` / `top_k_ragged_transform`(带 `dsa_graph_safe=True` 保证 CUDA graph 安全,row→batch 映射由 cu_seqlens_q 推导,decode-like 单行场景避免动态张量构造)。(3) torch backend 目前要求 `SGLANG_DSA_FUSE_TOPK=false`(仅非融合路径)。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py: 新文件,TopkBackend 枚举 + topk/topk_transform 分发 + flashinfer tie-break/deterministic 参数
python/sglang/srt/layers/attention/dsa_backend.py: DSAIndexerMetadata.topk_transform 委托 backend;fused page_table 选择加白名单
python/sglang/srt/server_args.py: --dsa-topk-backend 参数
test/registered/kernels/test_dsa_indexer.py: 新增 unfused 有效性 + fused 跨 backend 等价测试
```

## 性能声明

GSM8K platinum(1,209 题,8-shot,parallel 1209),DeepSeek-V3.2,kv-cache bf16,tp8/dp8 + dp-attention,moe flashinfer_trtllm_routed,dsa prefill/decode 均 flashmla_sparse,page 64(`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`,各 3 次运行):

| 配置 | Accuracy | Latency s | Output throughput token/s |
| --- | --- | --- | --- |
| sgl-kernel fused(基线) | 0.977 / 0.978 / 0.981 | 13.146 / 12.749 / 17.272 | 8566.7 / 8878.8 / 6584.3 |
| torch unfused | 0.982 / 0.983 / 0.980 | 18.256 / 17.637 / 17.609 | 6183.8 / 6389.0 / 6403.0 |
| flashinfer unfused | 0.978 / 0.978 / 0.979 | 20.846 / 24.896 / 21.313 | 5413.9 / 4557.0 / 5292.8 |
| flashinfer fused | 0.980 / 0.981 / 0.978 | 13.531 / 12.771 / 12.121 | 8320.2 / 8832.3 / 9267.3 |
| flashinfer fused tie_break=1 | 0.978 / 0.980 / 0.978 | 13.716 / 13.008 / 17.669 | 8219.6 / 8652.7 / 6457.7 |
| flashinfer fused tie_break=2 | 0.979 / 0.982 / 0.980 | 13.370 / 13.129 / 12.498 | 8438.6 / 8628.9 / 9047.7 |

即:flashinfer fused 吞吐与 sgl-kernel 基线相当,非融合路径(torch/flashinfer)明显更慢;正文声称的"长上下文性能更好"未附对应 benchmark。

## 验证与回归风险

新增两个单测:`test_topk_unfused_backends_valid_selection`(三 backend × tie-break × row_starts,验证 topk 有效性)、`test_topk_fused_backends_equivalence`(sgl-kernel vs flashinfer fused 输出排序后逐位相等,PAGED/RAGGED、多 query-len);报告的单元测试通过;CI Base/Extra 均通过。风险:torch backend 需关 fused topk(慢路径);fused 模式下 flashinfer/sgl-kernel 直接把 topk_indices 当 page_table 用,后端白名单之外会显式报错;tie-break/deterministic 行为差异会影响同分样本的选择(测试刻意用无 tie logits)。

## 关联

同区域(卡片库中):#7058(topk 后处理提速)、#17554(NSA/DSA 解码 metadata 优化)、#34167(top-k v2 修复)、#37591/#37941(ROCm DSA indexer top-k exact 化)、#39931(ROCm topk v2 split-k)。上游依赖:flashinfer#3095(可确定性/可 tie-break topk)。
