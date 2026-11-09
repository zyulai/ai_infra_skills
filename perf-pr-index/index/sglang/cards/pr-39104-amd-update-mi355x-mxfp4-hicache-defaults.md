# PR #39104: [AMD] Update MI355X MXFP4 HiCache defaults and quick-reduce quantization for Qwen3.5 cookbook

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ChangLiu0709 |
| 合并 | 2026-09-12 |
| 分类 | Quantization(major) |
| Label | documentation |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 为纯文档改动,覆盖 2 个文件(cookbook mdx 与 deployment jsx 生成器) |
| 链接 | https://github.com/sgl-project/sglang/pull/39104 |

## 一句话结论

更新 Qwen3.5-397B-A17B MXFP4 在 MI355X 上的部署配方默认值:HiCache 改 kernel/page_first、quick all-reduce 改 INT4,TPOT 降约 9%、交互性提升约 10%。

## 动机

MI355X 上 Qwen3.5 MXFP4 部署 cookbook 原有默认组合(hicache-io-backend=direct、page_first_direct、ROCM_QUICK_REDUCE_QUANTIZATION=INT8)在长上下文 agentic 与高并发场景下不是最优。上游 SemiAnalysis InferenceX #2693/#2972 已验证新组合(kernel/page_first + INT4 quick reduce)更快,PR 将 cookbook 与部署生成器同步到该验证过的配置,并刷新 MI355X Docker 镜像版本。本 PR 取代因分支重建冲突而关闭的 #35879。

## 优化点

无 kernel 代码改动,纯配置/文档调整,三处默认值变更:(1) HiCache IO backend `direct`→`kernel`、内存布局 `page_first_direct`→`page_first`,改变 host KV offloading 的读写路径;(2) ROCm quick-reduce all-reduce 量化 `INT8`→`INT4`,降低多卡 collective 的传输量(INT4 prefill allreduce 带来高并发下 +4.3% output tok/s/gpu 与 -13.6% P90 E2E latency);(3) MI355X 镜像从 `v0.5.15.post1-rocm720-mi35x-20260715` 升到 `v0.5.19-rocm720-mi35x-20260911`。同时更新 cookbook 文字描述使文档与生成器一致。

## 关键改动

```text
docs/src/snippets/autoregressive/qwen35-deployment.jsx: 部署命令生成器,HiCache 参数与 INT4 quick-reduce env 更新
docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx: MI355X Docker 镜像 bump 与 HiCache/quick-reduce 说明同步
```

关键改动片段(jsx 生成器):

```jsx
cmd += ' \\\n  --hicache-io-backend kernel';      // 原 direct
cmd += ' \\\n  --hicache-mem-layout page_first';  // 原 page_first_direct
...
amdEnv += "ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \\\n"; // 原 INT8
```

## 性能声明

PR 注明来源为 InferenceX #2972 的对比数据(未在本 PR 内附完整测试条件):

- kernel/page_first vs direct/page_first_direct:mean TPOT 8.2ms → 7.5ms(约降 9%),interactivity 121.7 → 134.1 tok/s/user(约 +10%),平台为 MI355X。
- INT4 quick reduce:prefill allreduce 在高并发下 +4.3% output tok/s/gpu,P90 E2E latency -13.6%。
- PR 正文未注明模型 batch/序列长度等 workload 细节(仅知是 Qwen3.5-397B-A17B MXFP4 MI355X 部署)。

## 验证与回归风险

纯文档改动,无数值路径风险;CI Base 测试通过,Extra 失败,AMD ROCm 无对应运行。风险主要是文档误导:若用户节点 CPU DRAM 不足 hicache-ratio=1.5 所需内存,或镜像 tag 失效,配方不可复现;quick all-reduce 与 AITER allreduce fusion 互斥这一约束需继续保持文档正确。

## 关联

被取代的 PR #35879;上游依据 SemiAnalysisAI/InferenceX #2693 与 #2972。同区域(AMD MI355X/quick-reduce)本卡未收录其他 PR,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
