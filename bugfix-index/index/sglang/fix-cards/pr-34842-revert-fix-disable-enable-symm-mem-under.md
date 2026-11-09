# PR #34842: Revert "[Fix] Disable --enable-symm-mem under CUDA graphs on Kimi hybrid models"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34842](https://github.com/sgl-project/sglang/pull/34842) |
| 作者 | wangfakang |
| 合并日期 | 2026-08-27 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | regression |

## 一句话结论
PyTorch 已上游修复 symm-mem 分配 bug(torch 2.13.0),删除 #33037 对 Kimi hybrid 强制关闭 --enable-symm-mem 的守卫。

## 问题现象
此前的守卫(#33037)在 Kimi hybrid 模型启用 CUDA graphs 时强制关闭 `--enable-symm-mem` 并打印警告;用户即使显式传参也会被忽略。正文说明其当初原因:PyTorch caching-allocator bug 使 per-forward 的 symm-mem 分配在不同 rank 落到不同 offset,捕获进 graph 后破坏 DSPARK spec decode——表现为 accept 塌缩到 1.000,或 server 静默输出垃圾且 accept 钉在上限;prefill 同样受累(同一分配位于被捕获的 RowParallelLinear 中)。该守卫在 PyTorch 修复后已无必要。

## 根因
正文披露:守卫针对的缺陷是 PyTorch caching-allocator 的 bug——symm-mem 分配按 per-forward 进行,被捕获进 CUDA graph 的地址既不在生命周期内保留、也不保证各 rank 同 offset,从而破坏 spec decode;该 bug 已在 PyTorch 上游修复(pytorch#183489),随 torch==2.13.0(#28836)提供。SGLang 现已运行在 2.13.0 上,守卫不再需要,故整体撤销 #33037。

## 修复方案
删除 #33037 加入的 disable 守卫,让 Kimi hybrid 在 CUDA graphs 下重新支持 `--enable-symm-mem`:
- `python/sglang/srt/arg_groups/kimi_k3_hook.py`:删除 `disable_kimi_k3_symm_mem` 函数(含按架构 `KimiLinearForCausalLM`/`KimiK3ForConditionalGeneration` 门控、declare_resolution 关闭 symm-mem 及警告文案的约 48 行)。
- `python/sglang/srt/server_args.py`:`_handle_cuda_graph_config` 中删除对应 import、过期注释与调用:
  ```diff
  -        from sglang.srt.arg_groups.kimi_k3_hook import disable_kimi_k3_symm_mem
  ...
  -        disable_kimi_k3_symm_mem(self)
           self._apply_cuda_graph_compatibility()
  ```
正文注明:无新增依赖/API,symm-mem 的 alloc/register 路径未动。

## 影响与回归风险
- 影响窗口:正文未说明;按 revert 对象,#33037 合并之后、本 PR 之前的提交受影响。
- 回归面:Kimi hybrid(KimiLinearForCausalLM/KimiK3ForConditionalGeneration)在 CUDA graphs 开启时的启动参数解析路径;`--enable-symm-mem` 在这些模型上重新生效。若运行环境仍为未含 pytorch#183489 修复的旧版 torch,原先被守卫挡住的 graph 捕获下 spec decode 损坏问题可能复现(依赖 torch>=2.13.0)。正文 CI 状态:Base 通过,Extra 与 AMD ROCm 7.2 失败。

## 关联
- https://github.com/sgl-project/sglang/pull/33037 (被 revert 的守卫 PR)
- https://github.com/pytorch/pytorch/pull/183489 (PyTorch 上游修复)
- https://github.com/sgl-project/sglang/pull/28836 (torch==2.13.0 升级)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
