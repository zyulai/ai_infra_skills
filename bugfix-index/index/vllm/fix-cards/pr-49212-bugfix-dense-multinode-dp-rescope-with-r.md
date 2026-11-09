# PR #49212: [BugFix] Dense multinode DP rescope with regression test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49212](https://github.com/vllm-project/vllm/pull/49212) |
| 作者 | aoshen02 |
| 合并日期 | 2026-08-06 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
修复多节点 dense 模型独立 DP rank 的端口冲突:新增 `reconfigure_for_independent_dp_rank()`,按 DP 组内重算节点数/节点序号,并补回归测试。

## 问题现象
正文未描述具体运行时报错现象。按 Purpose,dense 模型在多节点数据并行下每个独立非 MoE DP rank 被当作 DP=1 处理时,存在端口冲突(port conflicts);原始修复(#48626,由 Andy Lo 完成)因源 fork 被误转私有仓库而被 GitHub 自动关闭,无法重开,故本 PR 重建,并取代 #39405。

## 根因
正文未直接披露机制层面的根因分析。由 diff 推断:`run_engine_core` 与 EngineCore 的 `__init__` 中,非 MoE 独立 DP rank 原先只把 `data_parallel_size/_local/rank` 置为 1/1/0,而没有重设 `nnodes` 与 `node_rank`,导致多节点场景下各 DP rank 仍以原全局节点拓扑初始化(如端口分配),引发冲突;新 helper 先捕获 `nnodes_within_dp` 与 `node_rank_within_dp`,再把 DP 字段重置为单 rank 并以组内值覆盖节点字段。

## 修复方案
1. `vllm/config/parallel.py` 新增 `reconfigure_for_independent_dp_rank()`:
```diff
+    def reconfigure_for_independent_dp_rank(self) -> None:
+        """Reconfigure for a single independent non-MoE DP rank."""
+        # Capture these before changing DP fields.
+        nnodes = self.nnodes_within_dp
+        node_rank = self.node_rank_within_dp
+        self.data_parallel_size = 1
+        self.data_parallel_size_local = 1
+        self.data_parallel_rank = 0
+        self.nnodes = nnodes
+        self.node_rank = node_rank
```
2. `vllm/v1/engine/core.py` 两处(`run_engine_core` 与 EngineCore `__init__`)把原来手写的三行 `data_parallel_size/_local/rank` 赋值替换为调用该 helper。
3. `tests/test_config.py` 新增回归测试 `test_reconfigure_for_independent_dp_rank_on_multinode_dense_model`:构造 TP=8、DP=2、nnodes=2、node_rank=1 的配置,断言 rescoping 后 nnodes/node_rank 归约到 DP 组内(1/0)且 world_size=8。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:仅影响非 MoE 独立 DP rank 的配置 rescoping 路径(core.py 两处调用点)与 ParallelConfig 新增方法;DP 组内 nnodes/node_rank 语义变化会影响依赖这两个字段做端口/进程分配的逻辑。验证方面:#48626 关闭前历史 checks(DCO、pre-commit、文档、Buildkite 全矩阵)通过;本 PR 的回归测试因宿主机缺少编译好的 `torch.ops._C.silu_and_mul_quant` 自定义 op,在 collection 阶段未能运行。

## 关联
- #48626(被自动关闭的原 PR)
- #39405(被本 PR 取代的 open PR)
(正文以上述编号引用,未给出完整链接。)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
