# PR #28283: Fix inaccuracies and add NPU constraints in ascend_npu_profiling.mdx.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28283](https://github.com/sgl-project/sglang/pull/28283) |
| 作者 | qinsir5522 |
| 合并日期 | 2026-06-15 |
| labels | documentation, npu |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
修正 Ascend NPU profiling 文档中的多处不准确描述(trace 目录结构、/start_profile 时序、activities 支持范围),补充配置优先级说明与 `analyse()` 覆盖数据警告,并调整 NPU 重定向 Note 的位置。

## 问题现象
正文未描述具体现象(Motivation/Modifications 为空模板)。从标题与 diff 可知:`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_profiling.mdx` 存在多处不准确内容,按文档操作会踩坑——例如 Method C(`bench_serving --profile`)下 trace 目录结构写错;Terminal 2 示例让人「立即」发请求,而 server 收到 `/start_profile` 之前发的请求不会被采集;`activities` 参数未说明 NPU 上 MEM/RPD 的实际行为;`analyse()` 会覆盖已有输出目录但文档未提示。

## 根因
正文未披露根因。(由 diff 推断)该文档部分内容与 Ascend NPU 上的实际行为不符:trace 输出目录结构、profiling 触发时序、`activities` 在 NPU 上的支持范围(仅 CPU/GPU;MEM 依赖 CUDA memory API 会被静默忽略,RPD 需要 ROCm 会报错)等描述有误或缺失;此外 NPU 的 `torch_npu._apply_patches()` 重定向说明放在了 server 启动示例旁,而更适合放在「Collection Methods」节。

## 修复方案
纯文档改动,均在 `ascend_npu_profiling.mdx`:
1. 把「`torch_npu._apply_patches()` 自动把 CUDA activity 重定向到 NPU,`activities: ["CPU", "GPU"]` 实际采集 NPU 算子事件」的 Note 从启动示例处移到 Collection Methods 节;
2. 新增 Note:配置优先级为「API parameter > Environment variable > Default value」;
3. 修正双终端示例:Terminal 2 需等 Terminal 1 输出 "Waiting for N steps" 再发请求,并说明 server 收到 `/start_profile` 前发出的请求不会被捕获;
4. `activities` 说明改为「On Ascend NPU, only CPU and GPU are supported. MEM depends on CUDA memory APIs and will be silently ignored. RPD requires ROCm and will cause an error.」;
5. 修正 Method C 目录结构为 `<output_dir>/<timestamp>/<hostname>_<pid>_<timestamp>_ascend_pt/`;
6. 改写 merge_profiles 说明:NPU 上 merge 对 `*_ascend_pt` 格式支持不完整,合并结果可能不完整或不正确,建议逐节点查看 `trace_view.json`;
7. 新增警告:`ASCEND_PROFILER_OUTPUT` 已存在时运行 `analyse()` 会覆盖原目录,需要保留原始数据须先备份。

关键 diff 摘录:

```diff
-# Terminal 2: Immediately send inference requests to provide steps for profiling
+# Terminal 2: Wait for "Waiting for N steps" output from Terminal 1, then send requests.
+# Requests sent before the server receives /start_profile will not be captured.
```

## 影响与回归风险
- 回归面:仅文档(`docs_new/` 下 .mdx),不触及任何代码;零运行时回归。影响为文档读者照做后的行为预期:避免在 profiler 未启动时发请求、避免误用 MEM/RPD activity、避免 `analyse()` 覆盖原始数据。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
