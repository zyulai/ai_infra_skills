# PR #67041: [HiSparse] Revert the FlashInfer all-reduce backend override (#65201, #65522)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-11-13 |
| 分类 | Attention / Kernel(tier: major) |
| Label | nvidia, hisparse |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67041 |

## 一句话结论

撤销 #65201 与其后续 #65522 在 HiSparse 下强制 `trtllm` FlashInfer all-reduce backend 的 override:FlashInfer 0.7.4 已修复 MNNVL fused all-reduce + RMSNorm cluster-barrier race,HiSparse 恢复默认 backend 选择(单节点 `mnnvl`,正文/代码注释称 TP4/TP8 上约 3–5% throughput 优势),4xB200 复现 832/832 turns 零 CUDA error。

## 动机

#65201 为规避 MNNVL fused all-reduce + RMSNorm 的 cluster-barrier race(#64327),在 HiSparse 下强制 `trtllm` backend。该 race 的 FlashInfer 修复(flashinfer-ai/flashinfer#7782)已进入 FlashInfer 0.7.4(main 经 #67042 使用)并 cherry-pick 进 0.7.3;#65522 只是让该 override 保持在 lazy workspace 创建上,故一并撤销。`trtllm` override 使 HiSparse 损失单节点 `mnnvl` 的性能(注释:mnnvl 在单节点 NVSwitch 系统上 decode 快于 TRTLLM,H200/B200 TP4/TP8 约 3–5% throughput)。

## 优化点

- `_resolve_fi_ar_backend` 删除 HiSparse 特判(读 `get_current_vllm_config_or_none` 判断 `hisparse_config` 的整段逻辑移除),HiSparse 与其他负载一样走默认拓扑选择:单节点 `mnnvl`,多节点 Blackwell `mnnvl`(fallback 关闭),其他 `(None, False)`。
- `get_fi_ar_workspace` 移除 `resolved_backend` 参数(此前为 lazy 创建在 vLLM config 上下文外运行而传;现在 lazy 创建现场重新 resolve);`FlashInferAllReduce.__init__` 相应简化为 `backend, _ = _resolve_fi_ar_backend(self.group)`。
- 测试参数化删除 `hisparse` 维度(`(1, True, True) → ("trtllm", False)` 等 case 移除)。

## 关键改动

```text
vllm/distributed/device_communicators/flashinfer_all_reduce.py: 删除 _resolve_fi_ar_backend 的 HiSparse override 与 get_fi_ar_workspace 的 resolved_backend 参数
tests/distributed/test_comm_ops.py: test_flashinfer_auto_backend_uses_group_topology 参数化去掉 hisparse 维度
```

```diff
-    config = get_current_vllm_config_or_none()
-    # TODO: drop once FlashInfer ships flashinfer-ai/flashinfer#7782.
-    if config is not None and config.attention_config.hisparse_config is not None:
-        if node_count == 1:
-            logger.info_once(
-                "Using flashinfer allreduce backend trtllm: the mnnvl fused "
-                "all-reduce + RMSNorm can crash with HiSparse (vllm#64327)."
-            )
-            return "trtllm", False
-        logger.warning_once(
-            "FlashInfer allreduce is disabled: the mnnvl fused all-reduce "
-            "+ RMSNorm can crash with HiSparse (vllm#64327), and trtllm does not "
-            "support multi-node process groups."
-        )
-        return None, False
-
     if node_count == 1:
```

## 性能声明

- HiSparse 恢复默认 backend 选择后,单节点走 `mnnvl`;正文与所恢复的代码注释称其在单节点 NVSwitch 系统(H200、B200)上 decode 快于 TRTLLM,TP4/TP8 约 3–5% throughput(此数字为恢复默认路径的依据,非本 PR 新测的 E2E A/B)。

## 验证与回归风险

- 单测:`pytest tests/distributed/test_comm_ops.py -k "flashinfer_auto_backend or workspace_failure or standalone_workspace"` 8 passed。
- HiSparse 复现(#65522 的 4xB200 TP4 multi-turn repro,`host_pool_gib` 160,`VLLM_FLASHINFER_ALLREDUCE_BACKEND` unset 使每臂都走 `mnnvl`,同 server 与 evalscope 命令,每臂跑到首次崩溃为止):
  - `70eaf457c3`(#65522 baseline,lazy-workspace bug → `mnnvl`)+ FlashInfer 0.7.2.post1:2.5 分钟后 `CUDA error: unspecified launch failure`。
  - `70eaf457c3` + FlashInfer 0.7.3:832/832 turns,0 CUDA errors。
  - 本 PR(`b5c8e4011d`)+ FlashInfer 0.7.4:832/832 turns,0 CUDA errors;server log 确认 `backend=mnnvl`、`CUDA error` 计数 0;KV Cache Hit Rate 92%。
- `70eaf457c3` 两臂使用 #65522 的环境 shim(#61463 kernel fix,已合并,与 `os.memfd_create` polyfill);本 PR 臂只用 polyfill。
- 风险:依赖 FlashInfer ≥ 0.7.3(cherry-pick)或 0.7.4;老 FlashInfer(如 0.7.2.post1)在 `mnnvl` 路径下仍会触发 #64327 race。

## 关联

- #65201、#65522(被撤销);#64327(MNNVL fused all-reduce + RMSNorm cluster-barrier race);flashinfer-ai/flashinfer#7782(上游修复);#67042(main 升级 FlashInfer 0.7.4);#61463(kernel fix,env shim 组成部分)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
