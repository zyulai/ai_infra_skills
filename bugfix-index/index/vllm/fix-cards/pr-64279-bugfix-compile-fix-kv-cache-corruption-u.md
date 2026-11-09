# PR #64279: [Bugfix][Compile] Fix KV-cache corruption under whole-graph compile (revert #42129)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64279](https://github.com/vllm-project/vllm/pull/64279) |
| 作者 | ericschreiber |
| 合并日期 | 2026-10-29 |
| labels | bug, ready, torch.compile |
| 热度 | 👍 0 · 💬 4 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
#42129 把 vllm 自定义 op 加入 Inductor fallback allow-list,绕过了 PyTorch 默认布局约束,整图编译下 KV cache key 被按 head-major 步长写入而全部错乱;回退 allow-list 并对 torch<2.13 保留防挂死补丁。

## 问题现象
在 Transformers backend 模型(如 Qwen3、Kolibri-1 等)上,使用 compile mode 1/2,或 mode 3 且 `splitting_ops: []`(整图编译)时输出退化(degenerate output):main 分支上回归测试失败,`Matched tokens: []`,参考输出为正常文本而测试输出是 `>>1\n\n\n\n...` 之类乱码;GSM8K(Qwen3-0.6B,1319 题,5-shot,greedy)准确率从 eager 的 0.391 跌到 0.011、invalid 从 0.3% 升到 10.8%。默认 mode 3 不受影响。实现 issue #62865 的"方案 2",是 #62924 的回退式替代(与维护者及 #42129 作者在 Slack 讨论)。

## 根因
正文 Details 披露:#42129 把所有 `vllm::*` 与 `vllm_aiter::*` op 加入 Inductor 的 `FALLBACK_ALLOW_LIST`。Inductor 为 allow-list 内的 op 创建 fallback 时**不带 PyTorch 对自定义 op 的默认布局约束**;于是整图编译下 `vllm::unified_kv_cache_update` 收到 head-major 布局的 key,步长为 `(128, 128*S, 1)` 而非 eager 的 `(1024, 128, 1)`。`reshape_and_cache_flash` 只读 `key.stride(0)` 并假定 heads 打包存储,导致每个写入 KV cache 的 key 全部错乱,输出为垃圾。默认 mode 3 不受影响,因为它在 attention 与 KV-cache op 处切图,不走整图 fallback 路径。

## 修复方案
1. `vllm/env_override.py`:删除 `_VllmFallbackAllowList` 代理与 `_patch_inductor_fallback_allow_list`(即回退 #42129 的 allow-list),恢复 PyTorch 对所有 `vllm::`/`vllm_aiter::` 自定义 op 的默认处理(含默认布局约束)。
2. 保留 #42129 想解决的编译挂死防护(allow-list 当初只是为了跳过 torch<=2.12 会 eager 构建的 implicit-fallback INFO 消息,该消息对深 MoE/TP 图会挂死编译):新增 `_apply_inductor_operator_str_patch`,patch `OperatorIssue.operator_str`——对 `vllm::`/`vllm_aiter::` op 只返回 `f"  target: {target}"`(仅 op 名,不递归 stringify IR 参数);torch>=2.13(pytorch/pytorch#185951 已把消息惰性化)不应用 patch,通过 meta-path finder 延迟到 `torch._inductor.exc` 导入时再打,避免 import vllm 就拉起 inductor。
3. 测试:删除旧的 `test_inductor_fallback_allow_list_patch.py`,新增 `tests/compile/test_inductor_operator_str_patch.py`(验证 vllm:: op 的消息只含 op 名、IR 节点不被 stringify)与 `tests/models/transformers/test_backend.py::test_whole_graph_compile`(Qwen3-0.6B Transformers backend,mode 2 vs eager,`check_logprobs_close`)。

## 影响与回归风险
- 影响窗口:#42129 引入 allow-list 之后的版本,compile mode 1/2 或 mode 3 `splitting_ops: []` 的 Transformers backend 整图编译(正文未给具体版本号)。
- 回归面:所有 torch.compile 下 vllm 自定义 op 的 Inductor lowering 路径恢复 PyTorch 默认行为——非 vllm 命名空间不变;torch<2.13 平台(正文称目前只有 gfx1250 preview image)保留"消息只打印 op 名"的挂死防护,vllm op 的 lowering 错误信息在这些版本上只有 op 名。torch>=2.13 且开启 INFO graph logger 时深图上完整消息仍可能慢(`IRNode.__str__` 无深度限制,#43111),属 opt-in 调试日志。验证:GSM8K mode 2 0.401 恢复到与 eager/原生一致;CUDA 冷缓存编译时间在噪声内不变(Kolibri-1/Qwen3-4B 数据);ROCm 上 Kimi-K2 编译速度由 @djramic Slack 报告大致持平,作者本人无 ROCm 验证。后续建议:让 KV-cache writer 对 heads 不打包的 key 直接报错,避免未来布局变化静默腐蚀缓存;与 #57310 的 allow-list 改动重叠、应变得不再必要。

## 关联
- #62865(Fixes,实现其方案 2)
- #42129(revert 对象)
- #62924(回退式替代方案)
- #43111(IRNode.__str__ 无深度限制)
- #57310(重叠 allow-list 改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
