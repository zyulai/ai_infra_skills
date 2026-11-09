# PR #48045: [Bugfix] Fix FlashMLA dense fp8 metadata crash (num_sm_parts clamp)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48045](https://github.com/vllm-project/vllm/pull/48045) |
| 作者 | MatthewBonanni |
| 合并日期 | 2026-07-10 |
| labels | bug, ci/build, ready |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
FlashMLA dense fp8 元数据的 num_sm_parts 可算成 0 触发断言崩溃,上游 clamp 到 >=1 修复。

## 问题现象
修复 vllm-project/vllm#47935。DeepSeek-R1(128 heads)在 132-SM GPU(H100/H200)上使用 fp8 KV cache + FlashMLA 时,任意 67–128 token 的 prompt 都会崩溃——短 prompt(query 长度 ≤ `reorder_batch_threshold == 128`)被路由进 decode 路径,`num_q_tokens_per_head_k = max_query_len * num_q_heads`,在元数据计算处命中断言崩溃。正文测试环境为 H200(132 SMs)。

## 根因
正文披露:FlashMLA 的 `get_mla_decoding_metadata_dense_fp8` 计算

```
int num_sm_parts = sm_count / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
```

当 `num_heads_k == 1` 且 `ceil_div(num_q_tokens_per_head_k, block_size_m) > sm_count` 时结果为 0,触发断言;而 bf16/fp16 版本对最小值 clamp 到 1(fp8 dense 版漏掉了该保护)。

## 修复方案
修复落在上游 vllm-project/FlashMLA#14:对 `num_sm_parts` 与 bf16/fp16 版本一样 clamp 到 `>= 1`。vLLM 侧改动仅是 `cmake/external_projects/flashmla.cmake` 中 FetchContent 的 GIT_TAG 提升到含修复的提交:

```diff
-        GIT_TAG a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b
+        GIT_TAG b70aff3d110a2b1a037e62eac295166b5143643a
```

正文验证(打补丁后的 FlashMLA,H200):直接调用 `get_mla_decoding_metadata_dense_fp8`(num_q_tokens_per_head_k=8449 与 16384,此前均 `num_sm_parts → 0` 断言)返回有效元数据不崩溃;端到端在 dummy DeepSeek-V3(128 heads、1 个 dense 层、`--kv-cache-dtype fp8`)上 generate,100 与 128 token prompt 正常完成;66 token 对照组(`num_sm_parts = 1`)不受影响。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围;现象限于 fp8 KV cache + FlashMLA dense decode 路径、132-SM 级 GPU 上的短 prompt(67–128 token)。
- 回归面:vLLM 侧只动 FlashMLA 依赖版本,拉入上游 clamp 修复。所有 FlashMLA dense fp8 decode 用户行为改变仅体现在此前必崩的极端形状上;clamp 后 `num_sm_parts` 至少为 1,与 bf16/fp16 版本语义对齐,风险由上游 FlashMLA#14 承载,vLLM diff 本身无逻辑改动。

## 关联
- vllm-project/vllm#47935(被修复的 issue)
- vllm-project/FlashMLA#14(实际上游修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
