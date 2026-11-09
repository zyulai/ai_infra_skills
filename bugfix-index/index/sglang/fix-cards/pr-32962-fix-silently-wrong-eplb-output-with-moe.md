# PR #32962: Fix silently wrong EPLB output with --moe-a2a-backend none (rank-invariant dispatch)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32962](https://github.com/sgl-project/sglang/pull/32962) |
| 作者 | ch-wan |
| 合并日期 | 2026-07-31 |
| labels | documentation, deepseek, run-ci, bypass-fastfail |
| 热度 | 👍 1 · 💬 8 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

EPLB 在 no-a2a MoE 路径上各 EP rank 对 logical→physical replica 的选择不一致,冗余专家被多个 rank 重复计算求和,输出静默变差(GSM8K 0.63→0.42);改为按 token 行做 rank-invariant 的 replica 选择。

## 问题现象

EPLB 在 no-a2a MoE 路径(`--moe-a2a-backend none`,默认)上无任何报错地运行,但只要有多余的 redundant experts 需要放置,输出就被静默损坏。正文实测:`lmsys/sglang-ci-dsv3-test` 加 `--tp 2 --ep-size 2 --enable-eplb --ep-num-redundant-experts 48`,GSM8K 从约 0.63 跌到 0.42——没有 error、没有 warning。EPLB 在 0 redundant experts 时恰好正确(每个 logical expert 只有一个 replica,天然 rank-invariant),所以此前从未被发现。

## 根因

正文披露:没有 a2a backend 时,每个 EP rank 对**相同**的 tokens 运行 MoE、各自只持有物理专家的一个切片,partial outputs 相加求和,因此 logical→physical 的挑选必须在每个 rank 上完全一致。两处设计破坏了这一点:

1. `_compute_logical_to_all_physical_map` 把每个 logical expert 的候选列表塌缩为「距当前 rank 最近的 replica」——这在构造上就是 per-rank 的。所有 dispatch 算法都读这张塌缩后的 map,冗余 replica 因此既不可达(死重),又在 rank 之间不一致。
2. `static`(EPLB 下自动选中的算法)再从 per-rank 的表里挑 per-rank replica。被复制的 logical expert 便在多个 rank 上各执行一次、其输出被求和多次。

## 修复方案

不新增 `--ep-dispatch-algorithm` 取值:

- `dynamic` 在 `dispatch_must_be_rank_invariant(server_args)`(正文给出谓词为 `moe_a2a_backend == "none" and ep_size > 1`,经 `ExpertLocationDispatchInfo.rank_invariant` 传入)成立时,改为按 **token 行**选 replica(`arange // slots_per_token % num_valid`)——这是 token 位置的纯函数,各 rank 无需通信即一致;连续行仍交替 replica,保持 EPLB placement solver 假定的均匀负载切分。有 a2a backend 时保留原 `torch.randint` 随机选择(各 rank 只 dispatch 自己的 tokens,不一致无害;`static` 在该路径故意不一致以就近路由)。
- `_compute_logical_to_all_physical_map` 在 rank-invariant dispatch 下跳过 rank-local collapse,使冗余 replica 真正可达。
- `dynamic` 成为 no-a2a 路径的默认;`static` 与 `lp` 在该路径直接抛错拒绝(不再静默劣化);`fake` 与 `dynamic` 共用实现不受影响。a2a backend 行为完全不变(`static` 仍是默认,已用 `--moe-a2a-backend deepep` 验证)。

关键摘录(expert_location_dispatch.py):

```diff
-    chosen_dispatch_index = (
-        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
-        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
-    )
+    num_valid = info.partial_logical_to_all_physical_map_num_valid[topk_ids]
+    if info.rank_invariant:
+        ...
+        chosen_dispatch_index = row_index % num_valid
+    else:
+        chosen_dispatch_index = (
+            torch.randint(0, 65536, ...) % num_valid
+        )
```

正文解释不保留随机抽取的理由:今天它能工作只靠 seeds 广播与 DP padding 使各 rank 的 philox offsets 对齐,而没有任何断言保证;且不可复现——同一 greedy 请求随机抽取下 6 次出现 2 种输出,行索引下 6 次仅 1 种。验证:新增 nightly e2e 测试 `test_eplb_no_a2a.py`(两个变体分别覆盖初始放置与 ~70 次 rebalance);2×H200 GSM8K 200 题,本分支 0.625–0.66,恢复 collapse 为 0.42、完整修复前状态为 0.415/0.435;单测 35 通过。正文同时声明两个相邻问题(DeepseekV2MoE 非 a2a forward 以 enable_eplb 门控 dispatch info;`--ep-num-redundant-experts` 单独使用时不选 dispatch 算法)明确 out of scope。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,`--enable-eplb`(或非 trivial `--init-expert-location`)且 `--moe-a2a-backend none` 的部署默认算法由 `static` 变为 `dynamic`,且显式使用 `static`/`lp` 会启动报错;`expert_location.py` 的候选 map 构建逻辑在该路径被跳过 collapse。有 a2a backend 的路径不变。性能中性(一个 `arange`+整除替换同尺寸 `randint`)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
