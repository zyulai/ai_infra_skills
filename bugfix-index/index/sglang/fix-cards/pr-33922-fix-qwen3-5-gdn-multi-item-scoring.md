# PR #33922: Fix Qwen3.5 GDN multi-item scoring

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33922](https://github.com/sgl-project/sglang/pull/33922) |
| 作者 | daii-0818 |
| 合并日期 | 2026-09-10 |
| labels | run-ci, jit-kernel, run-ci-extra |
| 热度 | 👍 0 · 💬 34 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论

Qwen3.5 GDN prefill 把 packed 后缀当连续递归序列,后一个 item 继承前一 item 的 conv/SSM 状态;改为两阶段按 item 分支修复。

## 问题现象

Qwen3.5 hybrid 模型中,recurrent GDN 层与 full-attention 层混合。multi-item scoring 已给 full-attention 层提供 item 分支掩码,但 GDN prefill 路径此前把 packed 后缀当作一条连续递归序列处理,导致后面的 item 继承前面 item 的卷积(convolution)和 SSM 状态,跨 item 状态串扰。正文附复现启动命令(`--enable-mis --attention-backend flashinfer --linear-attn-prefill-backend triton` 等,模型 `Qwen/Qwen3.5-4B`)。

## 根因

正文披露:GDN prefill 路径此前未按 multi-item 分支切分,把整个 packed 后缀作为一条连续 recurrent 序列跑完,conv 与 SSM 状态沿序列顺序传递,后一个 item 的初始状态来自前一个 item 的结束状态,而非(应有的)各自 query 段结束状态。

## 修复方案

按正文 Modifications 与 diff:

1. Triton chunk gated-delta-rule 路径新增只读 final-state 模式:`chunk.py`/`chunk_delta_h.py` 增加 `inplace_update` 参数(默认 True 保持原地写回,False 时 kernel 不把最终状态写回 `initial_state`);XPU/CPU/NPU chunk kernel 不支持时显式抛 `NotImplementedError`。
2. `gdn_backend.py` 每次前向构建一次 GDN 专属 MIS 元数据(`GDNMISMetadata`,msgspec frozen struct,含 query/item 的 token 索引、cu_seqlens、request 索引),并校验分隔符严格递增、末尾分隔符为最后一个 token 等。
3. GDN MIS prefill 两阶段执行(`_forward_extend_mis`):先每个 request 计算 query 段(inplace_update=True 写回 query 末状态),再每个 item 从 query 末的 conv/SSM 状态分支(`_forward_mis_segments`);item 递归对 request SSM 池只读(inplace_update=False),conv 则为每个 item 使用独立克隆的槽位(`item_conv_states = conv_states[item_ssm_indices]`)。
4. 拒绝不支持的组合:非 Triton GDN prefill、page-major 布局、cached prefix、非 prefill-only 请求、target verify,以及未声明 `supports_mis` 的 hybrid linear backend(`MambaAttnBackendBase.validate_mis_support` 在构造时校验)。

关键 diff 摘录:

```diff
-        INPLACE_UPDATE=True,
+        INPLACE_UPDATE=inplace_update,
...
+        if self.mis_metadata is not None:
+            if is_target_verify:
+                raise ValueError("GDN MIS does not support target verify")
+            return self._forward_extend_mis(
```

验证(H20,Qwen3.5-4B BF16):MIS 端到端 3 项通过;专项套件 27 tests/46 subtests 通过;recurrent+causal-conv 全量 275 tests 通过;multi-item 分数与逐 item MIS 打分在 atol/rtol 2e-2 内一致。性能:multi-item scoring 对比独立 pointwise scoring,吞吐 +18.84%(52.80→62.75 items/s/GPU),P50 -16.04%、P99 -14.36%。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:GDN/chunk gated-delta-rule kernel 新增参数默认 True,常规路径行为不变;`hybrid_linear_attn_backend.py` 构造时新增 MIS 支持校验,`--enable-mis` 配不支持的 backend 会从"静默错误结果"变为启动报错;gdn_backend 的 forward_extend 在存在 MIS 元数据时改走新路径(仅 Triton prefill、无缓存前缀、prefill-only 场景),其余路径保持原逻辑;测试 kit 与单测同步扩展。
- **diff 截断**:pack 的 diff 在 40000 字符处按文件粒度截断(截断标记注明),未能见到的后续文件(如 Qwen3.5 模型侧改动及其余测试)未纳入本卡;本卡仅覆盖截断前可见改动。

## 关联

- Issue #31969(正文 Fixes #31969)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
