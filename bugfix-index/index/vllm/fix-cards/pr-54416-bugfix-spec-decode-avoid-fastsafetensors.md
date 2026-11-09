# PR #54416: [Bugfix][Spec Decode] Avoid fastsafetensors deadlock for PP draft models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54416](https://github.com/vllm-project/vllm/pull/54416) |
| 作者 | gcanlin |
| 合并日期 | 2026-09-12 |
| labels | bug, speculative-decoding, ready, mrv2, dflash |
| 热度 | 👍 2 · 💬 13 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
PP>1 且用 fastsafetensors 时,外部 draft 模型(仅存在于最后一个 PP stage)加载因 WORLD 组集体通信死锁;draft 加载自动回退标准 safetensors loader 修复。

## 问题现象
在 pipeline parallelism(PP>1)下使用 speculative decoding 且加载格式为 `fastsafetensors` 时,draft 模型加载死锁/失败:外部 draft 模型只在最后一个 PP stage 上实例化,而 `fastsafetensors` 目前使用 WORLD-group 广播,该 stage 上的 rank 会等待永远不会进入 draft loader 的其他 rank。影响面:PP + 外部 draft(EAGLE/EAGLE3、DFlash、DSpark 等)+ fastsafetensors 的组合。正文还指出替代方案(用 stage 的 TP 子组)也不可行:当前安装的 `fastsafetensors` PyTorch adapter 把子组内的 local source rank 当 global rank 传给 `torch.distributed`,在第二个 PP stage 立即报 `ValueError: Global rank 0 is not part of group`。e2e 验证环境为 2 节点 × 8 B200、Kimi-K3 + DSpark draft。

## 根因
正文披露:`fastsafetensors` 使用 WORLD-group 集体通信做加载广播,而 PP>1 时外部 draft 模型只实例化在最后一个 PP stage,该 stage 之外的 rank 不参与 draft 加载,导致等待方永远等不齐,形成死锁;改用 stage TP 子组又因 adapter 将子组 local rank 误当作 global rank(`ValueError: Global rank 0 is not part of group`)而不可行。因此修复思路是:draft checkpoint 远小于 target checkpoint,只为 draft 选择无集体通信的标准 safetensors loader,target 保持 fastsafetensors。

## 修复方案
- 新增 `vllm/v1/worker/gpu/spec_decode/utils.py::get_pp_safe_draft_load_config`:PP world_size>1 且 `load_format == "fastsafetensors"` 时,打 warning 并返回 `replace(load_config, load_format="auto")`,否则原样返回:
  ```python
  if get_pp_group().world_size > 1 and load_config.load_format == "fastsafetensors":
      return replace(load_config, load_format="auto")
  return load_config
  ```
- 在三处 draft 模型加载点接入:`dflash/utils.py::load_dflash_model` 与 `dspark/utils.py::load_dspark_model` 直接传 `load_config=get_pp_safe_draft_load_config(vllm_config.load_config)`;`eagle/utils.py::load_eagle_model` 中若返回的 config 与原 config 不同则 `replace(vllm_config, load_config=draft_load_config)`。
- `LoadConfig` 为复制修改,target 的加载配置不变;PP=1 与其他 load_format 行为不变。
- 新增/更新单测(tests/v1/...test_spec_decode_draft_load_config.py 等,39 passed)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围(Fixes #50959 所跟踪的问题)。
- 回归面:仅影响「PP>1 + fastsafetensors + 外部 draft」路径下的 draft 加载格式选择(改为 auto→标准 safetensors),target 模型加载、PP=1、其他 load_format 均不变;draft 加载改用标准 loader 后权重张量与推理逻辑不变(正文声明,并以生成 sanity check 与非零 speculative acceptance 验证:1055 proposed / 623 accepted,59.1%)。两个 backend-override 测试因新增 `load_config` 字段与 `get_pp_group` patch 而更新。

## 关联
- [Fixes #50959](https://github.com/vllm-project/vllm/issues/50959)
- [#50514](https://github.com/vllm-project/vllm/pull/50514)(正文提及:EAGLE3-style spec decode with PP,未修复本问题)
- [#53901](https://github.com/vllm-project/vllm/pull/53901)(正文提及:#50514 的 ROCm 扩展,未修复本问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
