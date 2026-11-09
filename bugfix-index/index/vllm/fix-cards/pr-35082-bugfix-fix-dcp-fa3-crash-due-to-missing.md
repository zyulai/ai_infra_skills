# PR #35082: [Bugfix] Fix DCP + FA3 crash due to missing num_splits in _forward_with_dcp

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35082](https://github.com/vllm-project/vllm/pull/35082) |
| 作者 | haosdent |
| 合并日期 | 2026-02-27 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 21 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
DCP 路径调 flash_attn_varlen_func 漏传 num_splits,与 capture 期 scheduler_metadata 形状不符致崩溃;补传对齐修复。

## 问题现象
Qwen3.5-397B-A17B(及类似模型)在 H200 GPU 上、FlashAttention 3 后端、`--decode-context-parallel-size 2` 下崩溃,报错:

```
RuntimeError: scheduler_metadata must have shape (metadata_size)
```

触发条件是 decode 上下文并行(DCP)+ FA3 的组合;正文未说明该组合的用户覆盖面,按指认凡是走 `_forward_with_dcp` 路径的 FA3 部署均受影响。

## 根因
正文披露:`_forward_with_dcp()` 调用 `flash_attn_varlen_func` 时漏传 `num_splits` 参数(取默认值 0),而 CUDA graph capture 期间 `get_scheduler_metadata()` 是以 `num_splits=32` 调用的。FA3 对 `num_splits=0` 与 `num_splits=32` 计算出的 `metadata_size` 不同(内部标志不同),于是前向阶段实际的 metadata 与 capture 期准备的 scheduler_metadata 形状不匹配,触发形状断言崩溃。即同一条注意力前向的两处调用点参数不一致。

## 修复方案
改 `vllm/v1/attention/backends/flash_attn.py` 的 `_forward_with_dcp`,给其中两处 `flash_attn_varlen_func` 调用补上缺失参数,与非 DCP 前向路径对齐:

```diff
             q_descale=q_descale,
             k_descale=k_descale,
             v_descale=v_descale,
+            num_splits=attn_metadata.max_num_splits,
         )
```

共两行、两个调用点,其余逻辑未动。需要说明:正文的 Test Plan 与 Test Result 章节为空,未提供测试命令或结果,修复验证情况以合并后 CI 为准。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;受影响的是 `_forward_with_dcp` 存在且未传 num_splits 的版本,DCP + FA3 组合必现。
- 回归面:按 diff 只给 DCP 路径的两处 flash_attn_varlen_func 调用显式传入 `attn_metadata.max_num_splits`,使前向与 capture 期 metadata 计算使用同一参数;非 DCP 路径与其他后端不受影响。风险点在于 DCP 路径的 split 行为从默认 0 变为与 metadata 一致的值,理论上改变 FA3 的 split-kv 调度,但与 capture 假设对齐本是原意;正文未提供 DCP 下的数值验证数据。

## 关联
- [#35057](https://github.com/vllm-project/vllm/issues/35057)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
