# PR #29304: [ROCm][PD] add moriio kv connector.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @inkcherry |
| 合并 | 2026-01-09 |
| 分类 | KV Cache / Memory \| tier: major |
| Label | documentation, rocm, frontend, ready, ci/build, v1, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)，pack 末尾标注 7 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/29304 |

## 一句话结论

MI300 上 MoRIIO PUSH 比 PULL 降 TTFT，32k 输入 2758→2614 ms。

## 动机

PR 引入基于 [MORI](https://github.com/ROCm/mori) 的 mori-io KV connector，用于 AMD 设备的 PD 分离 KV Cache 传输，支持 PULL 与 PUSH 两种模式。PR 正文说明目标是利用 MORI-IO 的 RDMA 传输最大化带宽利用率、最小化当前及未来 AMD 设备上的绝对传输时间，并通过 layer-wise push mode 在单请求内重叠计算与传输以降低 TTFT。借助抽象良好的 vLLM connector 接口，逻辑可集成到单个文件而不影响其他 core 文件。

## 优化点

集成 Mori backend，含 buffer merge、session cache management、batch io 组件。PULL mode 为 prefill/decode 串行交互；PUSH mode 为并行交互、非阻塞 layer-wise 传输：prefill 用专用线程维护传输队列，forward 正常进行的同时按层入队，收到 decode 的 block allocation 信号后异步调度传输；decode 先把已分配 blocks 发给 prefill 并进入等待队列，收到写完成信号后取出请求直接由 continuous batching 调度。PR 称该方式在 prefill 与 decode 之间无阻塞开销，兼容 decode graph mode 与 chunked prefill。提供统一 proxy example 覆盖 push/pull 逻辑，支持 xPyD；并行策略表为 TP/TP ✔️、DP/DP ✔️、TP/DP ❌。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py: 新增 MoRIIOConnector（1515 行），含 ZMQ 握手、notify、READ/PUSH 模式、TP/DP、session/offset 管理。
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py: 公共类型/配置/常量/模式与 zmq 上下文（321 行新文件）。
vllm/distributed/kv_transfer/kv_connector/factory.py: 注册 "MoRIIOConnector"。

  KVConnectorFactory.register_connector(
      "MoRIIOConnector",
      "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector",
      "MoRIIOConnector",
  )
```

PR 正文（Cursor Bugbot 摘要）还提及 `vllm/envs.py` 的 `VLLM_MORIIO_*` 开关、`examples/.../moriio_toy_proxy_server.py`、ROCm 安装文档、`docker/Dockerfile.rocm_base` 构建安装 MORI、`tests/v1/kv_connector/unit/test_moriio_connector.py` 单测；pack 有 7 个文件因体积被 dropped。

## 性能声明

测试条件：device MI300；Accuracy 配置 1P1D TP8、3P2D TP8、1P1D DP8、1P1D TP8(PULL MODE, fp8 kv cache dtype)、1P1D TP2(Qwen3-32B, non-MLA)，模型 DeepSeek-V3 / Qwen3-32B，gsm8k 5-shot；Performance 统一 `max_model_len 32k`、`max_num_batched_tokens 32k`、`kv_cache_dtype fp8`，`vllm bench serve` random dataset，ISL 2048/4096/8192/16384/32760、OSL 5、RATIO 0、CONCURRENCY 1、PROMPTS 1；PR 未注明模型权重精度。Accuracy（gsm8k exact_match flexible/strict）：1P1D TP8 0.9424±0.0064 / 0.8082±0.0108；3P2D TP8 0.9439±0.0063 / 0.8127±0.0107；1P1D DP8 0.9447±0.0063 / 0.8105±0.0108；1P1D TP8 PULL 0.9310±0.0070 / 0.8635±0.0095；1P1D TP2 Qwen3-32B 0.9439±0.0063 / 0.9325±0.0069。Performance（1P1D TP8 TTFT，PULL/PUSH ms）：2k 207/194、4k 340/311、8k 584/536、16k 1184/1090、32k 2758/2614。PR 正文未给出相对 baseline 的加速百分比，regex 未检测到定量 perf claim。

## 验证与回归风险

Accuracy 覆盖上述 5 组 gsm8k；Performance 覆盖 1P1D TP8 PULL/PUSH 在 2k~32k ISL 的 TTFT。Test Plan 给出 launch proxy（`moriio_toy_proxy_server.py`）、lm_eval gsm8k、各配置 `vllm serve` 与 `vllm bench serve` 命令；Bugbot 摘要提及单测覆盖 handshake、cache registration、metadata flow。Prerequisites 需安装 MORI（mori-ep 见 #28664）。风险与边界：并行策略表显示 TP(prefill)/DP(decode) 组合不支持（❎）；TODO 列出更多拓扑、chunk-wise prefill transfer、利用 prefill 首 token、降低 notify 开销；改动限定 AMD/ROCm 并经 factory 注册，不影响其他 connector；PR 未注明 CI 回归套件。

## 关联

PR 正文明确提及：#32204（同思路实现，但本 PR 去掉了 xfer 失败时的 kv blocks recovery）、#28664（mori-ep，依赖相同）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
