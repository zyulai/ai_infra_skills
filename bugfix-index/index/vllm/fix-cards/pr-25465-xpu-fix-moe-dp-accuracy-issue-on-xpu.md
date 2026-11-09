# PR #25465: [XPU] Fix MOE DP accuracy issue on XPU

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25465](https://github.com/vllm-project/vllm/pull/25465) |
| 作者 | faaany |
| 合并日期 | 2025-09-23 |
| labels | documentation, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
XPU 上 MoE DP 输出乱码:XpuCommunicator 缺 `dispatch`/`combine`,非 naive all2all 后端也不受支持;补齐方法、强制回退 naive 并允许关闭 expert parallel。

## 问题现象
XPU 上以数据并行运行 MoE 模型时输出完全乱码。用 `VLLM_WORKER_MULTIPROC_METHOD=spawn python examples/offline_inference/data_parallel.py --enforce-eager --model="ibm-research/PowerMoE-3b" --dp-size=2 --tp-size=2` 复现:修复前 DP rank 0/1 的生成文本是无意义重复串(如 `' a boatatatatatatatatatatatatatatatatatat'`、`' is is is is is is a0 to the the the the the the the'`);修复后两个 rank 均输出连贯文本。

## 根因
正文未写专门的根因分析,仅列明修复手段:为 `XpuCommunicator` 增加 `dispatch` 与 `combine` 方法(即此前缺失),并让 XPU 强制使用 `naive` all2all 后端。由 diff 推断:MoE DP 路径需要经由 communicator 的 dispatch/combine 完成 expert 并行的 token 分发与汇总,XPU 侧没有这两个入口,token/路由信息在各 rank 间错乱,导致精度崩坏;同时 `VLLM_ALL2ALL_BACKEND` 配成非 `naive` 值时 XPU 上既不支持也不拦截。

## 修复方案
`vllm/distributed/device_communicators/xpu_communicator.py`:
- 新增 `dispatch` / `combine` 方法,委托 `self.all2all_manager` 完成分发与合并;

```python
def dispatch(self, hidden_states, router_logits):
    assert self.all2all_manager is not None
    hidden_states, router_logits = self.all2all_manager.dispatch(
        hidden_states, router_logits)
    return hidden_states, router_logits

def combine(self, hidden_states):
    assert self.all2all_manager is not None
    return self.all2all_manager.combine(hidden_states)
```

- `__init__` 中当 `VLLM_ALL2ALL_BACKEND != "naive"` 时打 warning 并回退 `naive`(`naive` 成为 XPU 默认)。
`examples/offline_inference/data_parallel.py`:新增 `--disable-expert-parallel` 参数(默认 expert parallel 开启),并传入 `enable_expert_parallel=enable_expert_parallel`,替换原先写死的 `True`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:XPU 上的 MoE expert-parallel/DP 通信路径(dispatch/combine、all2all 后端选择)与 data_parallel 示例脚本的参数;非 XPU 平台不受影响。XPU 上显式配置非 naive all2all 后端的用户行为改变(告警并强制回退 naive);示例脚本默认行为不变,新增关闭 expert parallel 的开关。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
