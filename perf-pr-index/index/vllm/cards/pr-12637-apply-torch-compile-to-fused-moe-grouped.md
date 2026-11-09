# PR #12637: Apply torch.compile to fused_moe/grouped_topk

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-02-01 |
| 分类 | CUDA Graph / Compiler \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/12637 |

## 一句话结论

给 grouped_topk 与 DeepseekV3Model 加 torch.compile，DeepSeek-Coder-V2-Lite bs=1 生成 154→163 tok/s（+5%）。

## 动机

正文称其灵感来自 sgl-project/sglang 的 commit `1ebe1d6de5e0ce082e0be059c222baf0c5ee340a`，目的是改善 deepseek 模型的 generation latency——原因是这类模型 expert 数量很大。正文仅此一句说明动机，未展开更细的瓶颈分析。

## 优化点

在 `grouped_topk`（Deepseek-V2/V3 使用的分组 top-k 路由函数）上添加 `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)`，让该热点函数进入编译路径；同时在 `vllm/model_executor/models/deepseek_v3.py` 中为 `DeepseekV3Model` 启用 `@support_torch_compile`（此前是带 TODO 注释的被注释状态："TODO(simon): check whether we support torch compile for Deepseek V3"），并补上 `from vllm.compilation.decorators import support_torch_compile` 导入。两处都是编译开关类改动，不改动算法本身。

## 关键改动

- `vllm/model_executor/layers/fused_moe/fused_moe.py`：`grouped_topk` 增加 torch.compile 装饰器。
- `vllm/model_executor/models/deepseek_v3.py`：`DeepseekV3Model` 启用 `@support_torch_compile`，移除原 TODO 注释并新增导入。

```text
 # This is used by the Deepseek-V2 and Deepseek-V3 model
+@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
 def grouped_topk(hidden_states: torch.Tensor,
                  gating_output: torch.Tensor,
                  topk: int,
```

## 性能声明

条件：`DeepSeek-Coder-V2-Lite-Instruct`、bs=1、H100；脚本 `VLLM_MLA_DISABLE=1 python benchmarks/benchmark_latency.py --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --max-model-len 2048 --trust-remote-code --batch-size 1 --input-len 128 --output-len 512`。正文称 "we see an increase of 154 -> 163 tokens/s or 5% improvement at bs=1 on H100"。main：`Avg prompt throughput: 51.1 tokens/s, Avg generation throughput: 154.8 tokens/s`；this pr：`Avg prompt throughput: 51.2 tokens/s, Avg generation throughput: 163.8 tokens/s`。精度 PR 未注明测试条件；正文未给出其它模型、batch size 或硬件上的数据。

## 验证与回归风险

验证：正文给出 `benchmark_latency.py` 的 main / this pr 两段输出作为前后对比，PR 未注明新增单元测试或回归测试。可能受影响路径：Deepseek-V2 / V3 的 `grouped_topk` 路由、`torch.compile` 与 CUDA Graph / compilation 路径、不同平台的 `simple_compile_backend`；对其它模型或非 deepseek 路径正文未说明回归情况；正文也未给出 TTFT、显存占用或吞吐分项等指标，除 bs=1 外未注明其它 batch size 下的表现。

## 关联

正文仅提及灵感来源 sgl-project/sglang 的 commit（非 vLLM PR 编号），未给出同分类其它 vLLM PR 编号或前序/后续 PR，按 PR 编号口径写「无」。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
