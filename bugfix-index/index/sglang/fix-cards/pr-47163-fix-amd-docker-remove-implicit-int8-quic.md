# PR #47163: [Fix] [AMD] docker: Remove implicit INT8 QuickReduce default as it causes accuracy issues in 800k+ long context evals

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47163](https://github.com/sgl-project/sglang/pull/47163) |
| 作者 | functionstackx |
| 合并日期 | 2026-10-29 |
| labels | amd |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
ROCm 镜像把 `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` 设成所有工作负载的默认值,造成精度损失;删除该镜像级 ENV,改为用户显式 opt-in。

## 问题现象
QUICK REDUCE QUANTIZATION 被多名社区维护者与生产部署用户报告:终端用户生产工作负载以及私有/公开的 800k+ 长上下文评测质量显著下降。社区维护者与作者运行的一项评测中,800k+ 长上下文评测从 34% baseline 相对下降约 10% 到 30.4%。作者同时指出 AMD 自己也已在 vllm recipes 中注明过该选项的精度影响(Kimi-K3.yaml),因此影响精度的 flag 不应作为默认值。

## 根因
PR 正文披露:ROCm 镜像当前对每个工作负载都设置 `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`,即 INT8 QuickReduce 量化 allreduce 被隐式强制启用;这是一个影响精度的选项,不应设为镜像级默认,而应由部署方显式选择。

## 修复方案
从 `docker/rocm.Dockerfile` 中删除一行 ENV:

```diff
 ENV NCCL_MIN_NCHANNELS=112
-ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8
 ENV TORCHINDUCTOR_MAX_AUTOTUNE=1
```

正文说明:无其他默认值或运行时代码改动。当该变量未设置时,`QuickAllReduce.init_quick_all_reduce()` 默认取 `NONE` 并禁用 QuickReduce;用户仍可在容器启动时显式启用。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:使用 ROCm 镜像且未显式设置该变量的部署,QuickReduce 将从「隐式 INT8」变为「禁用(NONE)」。正文明确说明:未运行性能测试,移除隐式 INT8 QuickReduce 可能影响先前继承了该设置的工作负载的 collective 通信性能;作为交换,精度回到未量化 allreduce 的基线。

## 修复验证
- `git diff --check` 通过。
- 确认 diff 只包含所请求的 ENV 删除。
- 确认运行时 fallback 仍为 `NONE`。
- 未进行镜像构建或 GPU 精度测试。

## 关联
- [vllm recipes 中 Kimi-K3.yaml 对该选项精度影响的说明](https://github.com/vllm-project/recipes/blame/e38b528a540f3bf79674960cae78173e775e7fdc/models/moonshotai/Kimi-K3.yaml#L710)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
