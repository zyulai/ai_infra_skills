# PR #57797: [Bugfix][RL] Load attention sinks through weight_loader for layerwise reload

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57797](https://github.com/vllm-project/vllm/pull/57797) |
| 作者 | Ronald1995 |
| 合并日期 | 2026-10-14 |
| labels | bug, documentation, ready, qwen, deepseek, cpu, gpt-oss, DSv4, dflash, DSv4.1 |
| 热度 | 👍 0 · 💬 62 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
attention sink 写入绕过 weight_loader 回调,RL 分层重载后 sink 残留旧值;修复为统一注册 TP sink loader/factory 并让加载走注册回调。

## 问题现象
RL 权重更新过程中做 layerwise reload 时,attention sink 参数没有被刷新,残留旧值(stale sink values)。受影响的是所有带 attention sink 的模型加载路径:DeepSeek V4/V4.1、HY-V4、MiMo-V2(含 MTP 与 Omni MTP)、GPT-OSS、Laguna、Qwen3-DFlash。

## 根因
正文披露:attention sink 的写入此前绕过了参数的 `weight_loader` 回调——各模型在 `load_weights` 里用各自的临时逻辑(narrow + `param.data.copy_`)直接写 sink,不走参数上注册的 loader。分层重载走统一的 weight_loader 通道时,这些绕行的写入路径不会被触发,于是 sink 停留在旧值。

## 修复方案
- 在 `vllm/model_executor/model_loader/weight_utils.py` 新增通用实现:`make_attention_sink(num_heads, dtype)` 创建以 `-inf` 初始化、注册了 loader 的 sink 参数;`load_attention_sink` 按 TP rank 分片全局 checkpoint 的 heads、原地写本地分片,并把 padded heads 恢复为 `-inf`。loader 的分片长度取自 checkpoint 而非运行时参数(DeepSeek 的运行时 sink 有 padding,用 `sharded_weight_loader(0)` 会取错长度);checkpoint heads 必须能被 TP rank 整除。
- 删除各模型私有的 sink 处理分支,统一走注册回调:gpt_oss.py(mxfp4/quark/other 三条加载路径)、laguna.py(去掉 `_slice_sink`)、mimo_v2.py、mimo_v2_mtp.py、qwen3_dflash.py(去掉 `_preprocess`)、deepseek_v4 与 deepseek_v41 的 nvidia/amd/xpu 各 model.py/dspark.py/mtp.py 的 `attn_sink` 分支。
- GPT-OSS 保留 MXFP4、Quark 等量化分发;不再显式拒绝空 checkpoint;不兼容形状依赖张量切片与默认 loader。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:上述全部带 sink 模型的权重加载(含 TP 分片、padding 填 `-inf`、PP skip 路径)都改经 weight_loader;若某模型此前依赖「空 checkpoint 被显式拒绝」或 heads 不能整除 TP 的非标准 checkpoint,行为会变化。正文声明:未做全模型精度评估、真实 checkpoint 端到端 reload、TP>2 或多节点验证;backend 转换的 sink 副本与 GPT-OSS expert reload 的 follow-up 不在本 PR。pack 的 diff 在 40000 字符处截断(文件粒度),截断部分未见到。

## 关联
#57798, #57801, #57803, #57804, #57805, #57806(本 PR 整合并取代上述六个 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
