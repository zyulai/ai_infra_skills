# PR #54171: [Bugfix][ROCm][Build] fix profiler hang due to queue interposition bug

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54171](https://github.com/vllm-project/vllm/pull/54171) |
| 作者 | simondanielsson |
| 合并日期 | 2026-09-02 |
| labels | bug, ci/build, ready, rocm |
| 热度 | 👍 2 · 💬 21 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论

ROCm 镜像固定的 rocprofiler-sdk 缺少两个上游 queue interposition 修复,torch.profiler 启动即挂起;在 Docker 构建中打上这两个补丁。

## 问题现象

在 ROCm 上使用 torch profiler(`--profiler-config.profiler torch`)时 profiler 挂起,对应 issue #54087。PR 正文验证:用打了补丁的镜像 `vllm/vllm-openai-rocm:qi-fix` 启动 `vllm serve RedHatAI/gemma-4-31B-it-FP8-block-FP8-block`(load_format dummy),`/start_profile` 返回 200 后正常处理 completions 请求、`/stop_profile` 正常结束并输出 profiler 表格,不再挂起。

## 根因

正文说明:vLLM ROCm 镜像构建所固定的 rocprofiler-sdk commit(therock-7.14 release 一线,ROCm 7.2.3 构建)不包含两个上游修复——ROCm/rocm-systems 的 #7924 与 #7796,它们修复的正是 queue interposition 中的 bug,缺失导致 torch.profiler 在 ROCm 上挂起。正文给出的替代方案是升级到 rocprofiler-sdk 1.3.5,但那需要跟随 therock-10.0 一线,改动更大,故选择直接打补丁。

## 修复方案

只改 `docker/Dockerfile.rocm_base`:在拉取 rocprofiler-sdk 源码之后、编译之前,新增一段 `RUN`,按 commit 下载两个上游补丁并用 `git apply -p1 --include=projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/*` 仅应用到 hsa/queue interposition 相关文件:

```dockerfile
ARG ROCPROFILER_SDK_PR_7796=fae7fd57ffb8a76e811d6410b8a0cee1e94d710a
ARG ROCPROFILER_SDK_PR_7924=f8db22adbe17686a9b05ae03367dd1ec73011653
RUN cd /src && ... git apply -p1 --include="$HSA_PATHS" --verbose /tmp/qi-7796.patch \
    && git apply -p1 --include="$HSA_PATHS" --verbose /tmp/qi-7924.patch
```

Dockerfile 注释标明这是临时方案(TODO:待 `ROCPROFILER_SDK_COMMIT` 升到已含修复的版本如 ROCm 10.0 后移除)。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围,仅说明当前固定的 rocprofiler-sdk commit(therock-7.14 一线)受影响。
- 回归面:改动仅在 ROCm Docker 镜像构建阶段,对 vLLM Python 代码零影响。风险点在构建侧:补丁按 `--include` 限定 hsa 路径,若上游 rocprofiler-sdk 源码移动或 commit 变更,`git apply` 可能失败导致构建中断;补丁本身改变 queue interposition 行为,影响所有使用该镜像的 ROCm profiling 与运行路径。测试结果里 patch 应用干净(各 hunk 均成功,带 10~31 行偏移)。

## 关联

- https://github.com/vllm-project/vllm/issues/54087(修复的 issue)
- https://github.com/ROCm/rocm-systems/pull/7924(上游补丁)
- https://github.com/ROCm/rocm-systems/pull/7796(上游补丁)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
