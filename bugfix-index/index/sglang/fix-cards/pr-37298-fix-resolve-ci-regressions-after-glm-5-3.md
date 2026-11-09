# PR #37298: fix: resolve CI regressions after GLM-5.3 Flash rebase

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37298](https://github.com/sgl-project/sglang/pull/37298) |
| 作者 | Fridge003 |
| 合并日期 | 2026-09-01 |
| labels | apple-silicon |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
GLM-5.3 Flash rebase 后 CI 多点失败:修正 DSA MTP seed 宽度、保留 Lean Attention graph 元数据并修补测试与 GSM8K 门槛。

## 问题现象
#36507 rebase 之后的 post-rebase CI 出现多处失败(正文未逐条给出报错):GLM 的 MTP seed 宽度计算结果不对;多步 draft CUDA graph 元数据缺 Lean Attention workspaces;scheduler、HiSparse、MLX、FlashInfer hybrid、TRTLLM graph 的测试 fake 与当前 request/backend 契约不符;Lean/MLA paged-KV 旧测试仍用 4-D paged KV 视图;GSM8K 0.93 门槛对负载敏感——原 200 题/5-shot 高吞吐重跑在 0.930~0.905 间波动、低延迟 0.910,单题就值 0.5 个百分点。

## 根因
正文 Summary 披露:(1) MTP IndexShare seed 宽度计算不是架构中立的——GLM 包含 `index_kpool` 尾部槽,而 DeepSeek-V4 C4 保留未池化的 `index_topk` 宽度;(2) 多步 draft CUDA graph 元数据未保留 Lean Attention workspaces;(3) GSM8K 阈值只写在基类上会被 mixin shadow(mixin shadows values inherited from the server base),需放到两个具体 GLM-5.3 测试类;(4) 测试 fake 未跟上当前 request/backend 契约,统一内存暴露 dense 3-D KV 视图而旧测试仍构造 4-D paged 缓冲。

## 修复方案
- `model_config.py` 的 `get_dsa_mtp_topk_width` 改为直接读 `config.index_topk`:
```diff
-    return get_dsa_index_topk(config) + index_kpool - 1
+    return config.index_topk + index_kpool - 1
```
- `eagle_worker_v2.py` 仅在 `index_share_for_mtp_iteration` 为真时才计算 `dsa_seed_topk_width`
- `triton_backend.py` 的 `init_forward_metadata_out_graph` 传入 `lean_Mp/lean_Lp/lean_Op/lean_locks`,保留 Lean Attention workspaces
- GSM8K 阈值 0.93 与 500 题/20-shot 负载放到 LowLatency/HighThroughput 两个具体测试类
- Lean/MLA paged-KV 测试迁移到 dense 3-D KV 视图;更新各测试 fake 与 HiSparse/MLX 的 ReqKvInfo 构造

验证:EAGLE seed-width 单测 12/12,GLM-5.3 Flash 4x B300 500题/20-shot 低延迟 0.944、高吞吐 0.942。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:DSA/MTP seed 宽度计算路径(GLM 与 DeepSeek-V4 共用 `get_dsa_mtp_topk_width`);EAGLE v2 非 index-share 模型不再初始化 `dsa_seed_topk_width`;triton out-graph 元数据新增 lean_* 字段;多个 registered 单测的构造方式改变。

## 关联
- [#36507](https://github.com/sgl-project/sglang/pull/36507)(rebase 来源 PR)
- [验证 run 33456383043](https://github.com/sgl-project/sglang/actions/runs/33456383043)、[run 33457799782](https://github.com/sgl-project/sglang/actions/runs/33457799782)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
