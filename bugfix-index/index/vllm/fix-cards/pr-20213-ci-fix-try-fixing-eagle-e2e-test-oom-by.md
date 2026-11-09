# PR #20213: [CI Fix] Try fixing eagle e2e test OOM by reducing block allocation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20213](https://github.com/vllm-project/vllm/pull/20213) |
| 作者 | mgoin |
| 合并日期 | 2025-06-29 |
| labels | bug, ready, speculative-decoding |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | resource |

## 一句话结论
eagle e2e 正确性测试 CI 持续在 kv cache 分配阶段 OOM,测试内手工指定 GPU block 数与 max_model_len 绕过。

## 问题现象
`tests/spec_decode/e2e/test_eagle_correctness.py` 在 CI 中持续失败数周,失败表现为 kv cache 分配阶段的 OOM(对应 issue #20214)。作者不确定根因属于哪一类,只求先让 CI 过去。

## 根因
正文未确定根因,作者原话:「I'm not sure if it is a memory profiling issue or just a small GPU issue since the tests have failed for weeks at this point」——即不确定是内存 profiling 问题还是小 GPU 显存问题。采取的思路是手工指定 kv cache blocks 试试能否通过。

## 修复方案
不动产品代码,在两个 parametrize 的 `common_llm_kwargs` 中手工指定 block 数与最大长度:

```diff
+        # 2 for small prompt, 256//16 for generated.
+        "num_gpu_blocks_override": 2 + 256 // 16,
+        "max_model_len": (2 + 256 // 16) * 16,
```

分别应用于 `test_llama2_eagle_e2e_greedy_correctness` 与 `test_llama3_eagle_e2e_greedy_correctness`(即 2 个 prompt block + 256//16 个生成 block,block size 16)。Test Plan:看 spec decode 测试在 CI 是否转绿。

## 影响与回归风险
- 影响窗口:正文未写版本范围(仅说明测试已失败数周)。
- 回归面:仅测试配置;块数按 prompt+生成 token 硬编码估算,若测试的生成长度或 block size 变化需同步调整;产品代码路径无改动。

## 关联
[#20214](https://github.com/vllm-project/vllm/issues/20214)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
