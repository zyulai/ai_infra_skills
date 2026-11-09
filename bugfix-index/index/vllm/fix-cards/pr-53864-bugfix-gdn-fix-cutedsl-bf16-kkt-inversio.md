# PR #53864: [Bugfix][GDN] Fix CuteDSL BF16 KKT inversion divergence

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53864](https://github.com/vllm-project/vllm/pull/53864) |
| 作者 | Zoe923 |
| 合并日期 | 2026-09-17 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 15 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
CuteDSL GDN prefill 的 16x16 KKT 逆矩阵用 3 轮 BF16 Newton-Schulz 在相关 keys 下发散致输出错误;加一轮到 4 轮修复。

## 问题现象
Qwen3.5-397B-A17B-FP8 上,三轮 CuteDSL GDN prefill kernel 使确定性 greedy 生成相对 FLA backend 灾难性偏离:一个 9,991-token prompt 的真实业务请求在相同服务配置下返回空输出。数值上:确定性相关 key 案例输出最大绝对误差 `752.116699`;131 个真实 Qwen3.5 chunk × 32 heads 中最差 KKT 逆绝对误差 `98.5014`;CUTLASS DSL 4.5.2 与 4.6.2 均复现,非单一版本问题。修复后 200/200(C1-C8)与 2,000/2,000(C16-C200)响应与 FLA 逐字符一致,无空输出/乱码/Unicode 替换字符/超时/OOM/kernel 错误。

## 根因
正文披露:kernel 对每个 16x16 下三角 KKT 对角块用 3 轮 Newton-Schulz 求逆,中间逆矩阵每轮以 BF16 存回。3 轮在精确算术下对该块尺寸足够,但 BF16 舍入下,高度相关的 keys 会产生带瞬态增长的非正规 KKT 块,第 3 轮在 BF16 cancellation 收敛前结束,留下大的逆误差并传播进 GDN state 与输出。

## 修复方案
`vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_kkt_inv_uw.py`:Newton-Schulz 轮数 3→4:

```python
-                # 3 rounds of Newton-Schulz
-                for _ in cutlass.range_constexpr(3):
+                # The fourth round keeps correlated-key KKT blocks stable in BF16.
+                for _ in cutlass.range_constexpr(4):
```

选 4 轮为最小充分修复:最差逆误差 3 轮 `98.5014` → 4 轮 `0.043701`(5 轮 `0.0078125`,但均值误差几乎不再改善,4 轮为最优 correctness/performance 权衡);kernel 平均延迟 +0.38%,CuteDSL 仍比 FLA 快 2.09x-3.20x,端到端各并发点吞吐 +6.36%~+11.07%。新增确定性回归测试 `test_gdn_chunk_cutedsl_correlated_keys_stability`(64-token 高度相关 keys、`beta=0.9`、`g=0`、FP32 零初态),三轮实现下失败(`expected: < 0.1`),四轮下 7 passed。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:仅 CuteDSL GDN prefill kernel 的数值收敛;不改 GDN 数学定义、tensor 布局、metadata 或公共 API;FLA backend 不受影响。90 个真实隐藏态回放(45 层 × 2 TP rank)全部有限且过误差门(输出相对 L2 误差最差 0.238%)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
