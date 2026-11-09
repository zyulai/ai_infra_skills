# PR #26218: fix[DP][v1]: Prevent hangs from mismatched worker configurations

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26218](https://github.com/vllm-project/vllm/pull/26218) |
| 作者 | ayushsatyam146 |
| 合并日期 | 2025-10-08 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
DP leader 与 worker 的 EPLB 配置不一致时部分 worker 才参与 All-Reduce 导致挂死;把 EPLB 参数纳入 parallel config 哈希并在握手时校验,不一致快速失败。

## 问题现象
Fixes #26187:vLLM 在 DP 部署下,当 leader 与 worker 拥有不同的 EPLB 配置(例如 `--enable-eplb-log-balancedness`)时可能挂死(hang)。原因是这类配置差异导致只有部分 worker 尝试发起 All-Reduce 调用,其余 worker 不参与集合通信,等待方永远等不齐。正文未给出具体日志或影响版本。

## 根因
正文披露:leader 与 workers 的配置存在影响集合通信模式的差异(EPLB 相关配置),但此前的并行配置校验没有覆盖这些参数,不匹配的配置让部分 worker 走入 All-Reduce 调用而另一部分不走,通信参与方数量不一致从而挂起。即配置哈希(`parallel_config.compute_hash`)未包含 EPLB 参数,握手阶段也没有任何机制比对各 DP worker 的完整配置。

## 修复方案
三项改动:
1. `vllm/config/parallel.py` `compute_hash`:因子列表追加 `enable_eplb`,并在启用时追加 `eplb_config.log_balancedness`、`eplb_config.window_size`、`eplb_config.step_interval`、`eplb_config.num_redundant_experts`,使哈希覆盖 EPLB 配置;docstring 注明该哈希也用于 DP worker 配置校验。
2. `vllm/v1/engine/core.py` `_perform_handshake`:READY 消息在 `data_parallel_size > 1` 时附带 `parallel_config_hash = vllm_config.parallel_config.compute_hash()`。
3. `vllm/v1/engine/utils.py`:`EngineHandshakeMetadata` 新增 `parallel_config_hash: Optional[str]` 字段,init 消息带上 hash;`wait_for_engine_startup` 收到 READY 后校验:
```python
worker_config_hash = msg.get("parallel_config_hash")
expected_hash = parallel_config.compute_hash()
if worker_config_hash != expected_hash:
    raise RuntimeError(...)
```
错误信息明确指出"所有 DP worker 在影响集合通信的参数(如 enable_eplb、eplb_config.log_balancedness)上必须一致,请用相同的命令行参数启动",实现配置不匹配时 fail fast 而非挂死。

## 影响与回归风险
- 影响窗口:正文未说明(v1 引擎,labels 含 v1)。
- 回归面:DP 启动握手路径(`v1/engine/core.py`、`v1/engine/utils.py`)与 parallel config 哈希计算。所有 DP size>1 的启动都会多传一个 hash 字段并做一次字符串比对;`compute_hash` 输出变化可能影响依赖该哈希做持久化/缓存键的场景(正文未提及,按哈希参与因素增加如实陈述)。旧版本 worker 与新版本 leader 混布时,worker 不带 hash 字段(`msg.get` 返回 None)会触发 mismatch 报错,属于预期 fail-fast 行为。

## 关联
- #26187(Fixes)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
