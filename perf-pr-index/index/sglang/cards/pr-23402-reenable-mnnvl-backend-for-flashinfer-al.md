# PR #23402: Reenable MNNVL backend for FlashInfer allreduce fusion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenscarl |
| 合并 | 2026-06-16 |
| 分类 | Attention / Kernel |
| Label | documentation, lora, deepseek, run-ci, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/23402 |

## 一句话结论

重新启用 FlashInfer MNNVL fused allreduce+residual+RMSNorm:GB200 TP2 上 TPOT 较 trtllm 后端快 1.12-1.19x,并修复其与 piecewise CUDA graph 的挂死。

## 动机

FlashInfer unified fused allreduce+residual+RMSNorm 最初由 #12787 集成,因挂死在 #20792 被回退。挂死与 piecewise CUDA graph(PCG)capture/replay 期间 MNNVL 式 allreduce fusion 相关,栈指向 FlashInfer MNNVL 实现(`trtllm_mnnvl_allreduce.cuh` 约 557-573 行)Lamport 式等待循环无进展,与 oneshot/twoshot 的进度假设在 PCG replay 下冲突;eager 与 decode(非 piecewise)路径未复现。相关报告见 vLLM #35772。

## 优化点

重新引入 MNNVL 后端并以显式 flag 管理:新增 `--flashinfer-allreduce-fusion-backend`(auto|trtllm|mnnvl),fusion 门控改为 `backend is not None`,旧 `--enable-flashinfer-allreduce-fusion` 弃用并映射为 auto。后端解析:`mnnvl` 在任何 Blackwell(SM10x,单/多节点)或 SM90 单节点可用,`auto` 在支持处解析为 mnnvl、否则 trtllm;不支持的显式请求(如 SM90 多节点)回退 trtllm。核心安全策略:MNNVL + PCG 视为不安全——`mnnvl`/SM100 `auto` 时在 `model_runner.py` 直接禁用 piecewise CUDA graph,fused op 保留为 PCG split op 在 graph 片段间 eager 执行;`trtllm` 则在不需要 MNNVL split 时从 `split_ops` 移除,使 fusion 留在图内编译。其他:fusion 返回 `(None, None)` 时强制先做 `tensor_model_parallel_all_reduce` 再 RMSNorm(修复 fallback 丢 allreduce);workspace 改用 `create_allreduce_fusion_workspace(backend=...)` 统一创建,记录 token/hidden 高水位使 workspace 只增不重建;为 MNNVL workspace 初始化提供基于 torch.distributed 的 `TorchDistributedCommBackend`(替代 MPI)。自动开启策略收紧为仅 SM100(指定 MoE 模型、TP>1、单/多节点、无 dp attention),SM90 不再自动开启(缺 NVLink multicast fabric)。

## 关键改动

```text
python/sglang/srt/layers/flashinfer_comm_fusion.py: 后端解析/支持矩阵、MNNVL comm backend、workspace 统一创建与复用
python/sglang/srt/server_args.py: --flashinfer-allreduce-fusion-backend 参数与 auto-enable 策略(仅 SM100)
python/sglang/srt/model_executor/model_runner.py: MNNVL 下禁用 piecewise CUDA graph
python/sglang/srt/layers/communicator.py: fusion 门控改为按 backend 判断
test/registered/unit/layers/test_flashinfer_comm_fusion.py: 新增后端解析与 torch baseline 一致性单测
```

## 性能声明

- GB200 4 GPUs,gpt-oss-120b TP2,bench_serving random(ISL/OSL 1024/1024,128 prompts),MNNVL vs TRT-LLM(开 piecewise CUDA graph)TPOT 加速比:conc 1: 3.30 vs 3.79 ms(1.15x)、4: 1.12x、16: 1.14x、32: 1.17x、64: 8.93 vs 10.60 ms(1.19x);vs TRT-LLM 关 PCG 为 1.15x-1.21x。
- B300 SXM6 8x GPU TP8,ISL 4096/OSL 512,bench_one_batch_server:TPOT 2.04/2.26/2.98/3.51/4.21 ms(MNNVL)vs 2.17/2.36/3.09/3.67/4.57 ms(TRT-LLM),加速 1.04x-1.09x(conc 1/4/16/32/64)。
- 精度:GSM8k 200 题,B300 上 MNNVL 96.5%、TRT-LLM 98.0%(吞吐均 ~12,400 tok/s);另一组 Gb200 few-shot 运行记录 Accuracy 0.875 / Invalid 0.020。

## 验证与回归风险

新增 `test_flashinfer_comm_fusion.py`:按架构 mock 验证 auto 解析矩阵(Blackwell 多/单节点→mnnvl,SM90 多节点→trtllm,pre-SM90→trtllm),并用 fake comm 验证 trtllm/mnnvl 输出与 torch baseline 一致;LoRA e2e 测试加入 `flashinfer_allreduce_fusion_backend="trtllm"`。风险面:默认策略变化——SM90 上受支持 MoE 模型不再自动开启 fusion;MNNVL 与 PCG 组合被整体禁用(以 PCG 换 fusion,MNNVL 收益以失去 PCG 为代价);workspace 复用逻辑与 `(None, None)` fallback 行为变更影响所有走该 fusion 的模型;PR CI 显示 Base 失败(no_entry_sign)、Extra 通过。

## 关联

前序:PR #12787(原集成)、PR #20792(因挂死回退,本 PR 恢复);相关:vLLM #35772。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
