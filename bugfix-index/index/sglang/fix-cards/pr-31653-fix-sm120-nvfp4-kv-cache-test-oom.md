# PR #31653: Fix SM120 NVFP4 KV cache test OOM

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31653](https://github.com/sgl-project/sglang/pull/31653) |
| 作者 | Fridge003 |
| 合并日期 | 2026-07-18 |
| labels | blackwell |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Quantization |
| bug 类型 | resource |

## 一句话结论
SM120 Llama NVFP4 KV cache 测试把显存占满致 FlashInfer autotune 时启动 OOM;mem_fraction_static 降为 0.87 修复。

## 问题现象
新增的 SM120 Llama-3.1-8B-Instruct-NVFP4 + NVFP4 KV-cache 精度测试中,server 启动失败:FlashInfer autotuning 执行第一次 ModelOpt FP4 GEMM 时,CUDA 仅剩 28.44 MiB 空闲,而 FlashInfer 还要再申请 32 MiB,server 启动失败,测试 harness 随后报告 exit code `-9`。影响面仅限该新增 CI 测试(`extra-a` 阶段、`1-gpu-small` runner、32 GiB RTX 5090),正文明确这是窄范围的 CI 收敛,不涉及生产环境。

## 根因
正文披露:测试在默认 static-memory fraction(0.885)下把 NVFP4 KV-cache 分配拉到最大,留给运行期的空闲显存不足——FlashInfer autotune 申请 32 MiB 时 CUDA 只剩 28.44 MiB,启动即 OOM。

## 修复方案
1. 测试从 `test/registered/models_e2e/` 迁移到 `test/registered/quant/`(原文件删除、新文件同名创建)。
2. 结构由 `run_combined_tests` + `ModelLaunchSettings` 扫描改为 `GSM8KMixin` + `DefaultServerBase` 单一 server 配置。
3. 启动参数新增 `--mem-fraction-static 0.87`(默认 0.885),保留 200 并发,在 32 GiB RTX 5090 上多留约 0.47 GiB 运行余量:

```diff
+        "--page-size",
+        "64",
+        "--mem-fraction-static",
+        "0.87",
+        "--cuda-graph-backend-prefill=disabled",
```

4. 注册时声明 `disabled="Temporarily disabled due to failing accuracy"`,因该测试当前未过精度阈值,暂时停用;测试仅覆盖 NVFP4 GEMM + NVFP4 KV-cache 配置,不含 MTP 变体。

## 影响与回归风险
- 影响窗口:该测试为新增用例,失败 job 见正文链接;正文未说明涉及的生产版本范围。
- 回归面:仅测试代码与 CI 注册,不改变生产 allocator 行为(正文明确 "it does not change production allocator behavior");测试处于 disabled 状态,直到其独立的精度失败问题解决;精度阈值沿用实测 NVFP4 KV 分数 0.632420091324201 减 0.015。

## 关联
- 失败 job:https://github.com/sgl-project/sglang/actions/runs/29624032862/job/88035809117

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
