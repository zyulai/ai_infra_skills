# PR #33824: [Simulator] Add high-fidelity CPU-based inference simulator

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @littlefatfat |
| 合并 | 2026-09-04 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, dependencies, hicache, run-ci, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但超长截断(62 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/33824 |

## 一句话结论

新增 SGLang Simulator:在 CPU 上复用真实调度器/radix cache/请求生命周期,把 model forward 换成延迟预测器,免加载权重即可评估调度、缓存与吞吐配置(非加速 PR)。

## 动机

大规模推理的配置研究(调度策略、缓存层级、workload 形态)直接在加速器上跑代价高。若能在不加载模型权重、不执行 GPU kernel 的前提下复现 SGLang 的调度与 KV cache 行为,就能低成本做配置扫描与容量规划。本 PR 是 #22250 原型的 current-main 重写,兼容性基线锚定 SGLang v0.5.16(实测验证覆盖 v0.5.16/0.5.17/0.5.18 与 main)。

## 优化点

核心机制是"替换执行、保留系统":以 dummy 权重与最小 KV 分配启动真实 SGLang runtime(强制 tp=ep=dp=pp=1 单 worker、torch_native attention、disable overlap/cuda graph),把 model forward 替换为三种 forward 延迟预测器之一——AIConfigurator(算子/模块性能数据库)、sklearn 兼容的 18 特征 ML 模型、batch-composition replay(精确/最近邻查表)。支持 OFFLINE(逻辑时钟推进,不睡眠)与 BLOCKING(按预测的 forward 与 L2→L1 加载延迟真实睡眠)两种模式;replay 只替换 forward 延迟,CPU 前后处理默认保留。workload 侧支持 Random、ShareGPT 与带时间戳的 Autobench JSONL trace;执行侧支持进程内 `SGLangBenchmarkRunner` 与 HTTP server+client 两种路径;输出 request/latency/throughput/prefix-cache/cache-tier/predictor 指标(metrics.json、request.jsonl、iteration.jsonl)。调度器拓扑在 sim config 中"建模"而非实际拉起并行 worker。`tools/sglang-simulator` 变更时由 CPU PR CI 触发回归套件。

## 关键改动

```text
tools/sglang-simulator/: 新增 monorepo 组件(pyproject/README/示例配置与 workload)
src/sglang_simulator/compat.py: SGLang launch/serving API 兼容性检查与 simulator 专属 ServerArgs 覆盖
src/sglang_simulator/dataset/autobench.py: simulator 自有的 Autobench JSONL trace 解析
docs/docs/advanced_features/sglang_simulator.mdx: 用户文档(quickstart/指标读取)
```

```python
+SIMULATOR_SERVER_ARG_OVERRIDES = {
+    "tp_size": 1, "ep_size": 1, "dp_size": 1, "pp_size": 1,
+    "disable_overlap_schedule": True, "disable_cuda_graph": True,
+    "attention_backend": "torch_native", ...
+}
```

## 性能声明

本 PR 不含加速声明(regex 未检出性能 claim);PR 正文给出的是模拟器保真度(与实测 serving trace 对比的绝对百分比误差,TTFT/TPOT/ITL/吞吐/时长为 MAPE 型绝对误差,prefix reuse 为绝对百分点误差):

- Qwen3-8B(H20,AIConfigurator,no-cache/L1/L2):TTFT 2.38–4.15%,TPOT 1.64–4.05%,ITL 1.78–4.07%,输入吞吐 1.41–2.40%,时长 1.39–2.33%,prefix reuse 0.00–0.04 pp。
- Qwen3-32B-FP8(H20,AIConfigurator):TTFT 2.40–3.05%,TPOT 0.58–1.11%,其余 ≤1.13%。
- GLM-5.1-FP8 / DeepSeek-V4-Pro(v0.5.11,ML):TTFT 5.27% / 6.12%,输入吞吐 1.11% / 0.23%(TPOT/ITL 不适用,PD 分离单 token 输出)。
- DeepSeek-V4-Flash(v0.5.16,ML):TTFT 0.786–1.343%,吞吐/时长误差 ~0.0002%。
- v0.5.16 四条 32–128K 实测 trace(AIConfigurator):mean-TTFT MAPE 5.89%,时长与输入吞吐 MAPE 均 0.011%,prefix-reuse 最大误差 <0.011 pp。

## 验证与回归风险

- 维护测试:`test_simulation_sglang_runner.py`(进程内 runner)、`test_simulation_sglang_serving.py`(HTTP serving,AIC/ML/replay × ShareGPT/时间戳)、`test_simulation_offline_blocking.py`(两模式逻辑等价)、`test_simulation_cache_hit_ratio.py`(prefix 命中与 cache-tier 计量);工具目录变更时 CPU PR CI 自动跑回归。
- 风险与限制:prefix-cache 精度对 `max_total_tokens` 高度敏感,文档要求复制真实服务器日志的 `max_total_num_tokens`;模拟 runtime 恒为单 worker,tp/ep/dp CLI 参数被忽略(拓扑仅在 sim config 中建模);AIC 额外依赖被 pin 到 `aiconfigurator==0.10.0`;OFFLINE 模式下延迟/吞吐必须读 server-side 指标而非客户端墙钟。CI 显示 Extra 失败、Base 沙漏(未完成)、AMD 无运行。

## 关联

- 前序:#22250(模拟器原型)。
- 同区域:HiCache(hierarchical cache)行为模拟;与 perf-baselines 类基础设施互补(用于配置研究而非回归基线)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
