# PR #51052: [KVConnector][MoRIIO] Transfer hybrid mamba/KDA recurrent state in READ mode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YukioZzz |
| 合并 | 2026-09-20 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, rocm, ready, kv-connector, kimi, k3, kv-cache-manager |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 9 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/51052 |

## 一句话结论

MoRIIO READ 模式支持 hybrid attention + Mamba/KDA 模型的 PD 分离:conv/SSM 递归状态随 attention KV 一起传输,decode 不再从空状态启动。

## 动机

hybrid 模型在 disaggregated serving 中仅传 attention KV 不够:递归 conv 与 SSM 状态必须随 KV 一起移动,否则 decode 从空递归状态开始、输出损坏。MoRIIO 此前只覆盖纯 attention 组。范围刻意收窄:READ 模式、无 spec decode、生产/消费同构 TP;Hybrid WRITE 与异构 TP 递归状态 relayout 为 follow-up。

## 优化点

三个提交分层:①请求级传输完成语义(每层一个 status、KDA 层两个,聚合为 DONE/FAILED/PENDING);②把 packed `[B, 1, 1, C]` 字节页注册/寻址为独立 MoRIIO region——按 `MambaBase.bind_kv_cache` 相同的 squeeze→slice→view 契约解包,块在多层共享分配内 strided 时保持零拷贝;③per-group 状态 block ID 随调度携带并向 KDA 层发起 READ,KDA offset-template 缓存复用传输偏移。约束与语义:单一可传输 attention group、多个等价 Mamba group 保持独立不展平、异构 TP 显式拒绝;producer 停在 h(N-1),decode 本地重算最后一个 prompt token 推导 h(N);full attention 命中时也传输递归状态;READ 在 decode 分配前被中止则释放 producer block 且不开会话/不发 RDMA;失败在 forward 前 fail closed(路由协议不变,仍走 `remote_block_ids`)。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py: hybrid 调度、block group 拆分、READ 发起与中止语义
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py: TransferBatchState 聚合、recv_abort_timeout 配置
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py: KDA conv/ssm 区域与 mamba offset 模板
vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py: MambaConvSplitInfo 拆分信息
tests/v1/kv_connector/unit/test_moriio_hma_scheduler.py(新): 纯 CPU 单测(N-1 记账、prompt 截断幂等、offset 模板缓存)
```

## 性能声明

无加速比;正确性/规模验证:MI355X focused 单测 153 passed;Kimi-Linear-48B-A3B-Instruct 双 MI355X 节点(TP2/DCP2、fp8 KV、RDMA READ、无 spec)GSM8K N50 44/50 = 0.88,与既有基线一致;Kimi-K3 TP8/DCP8 AITER GSM8K full 1279/1319 = 0.96967,0 错误;AgentX 生产栈 1 小时 c40:2019 有效请求,输入吞吐 70,236.35 tok/s、输出 478.10 tok/s,TTFT p50/p99 2.90 s / 160.88 s,理论 prefix 命中 96.73%、实测 92.18%。

## 验证与回归风险

5 个 moriio unit 文件(权威运行在 ROCm + mori 的 MIA recipe 镜像);不支持场景显式失败(WRITE、speculative、异构 TP、不兼容 Mamba spec、非法 packed layout)。风险:scheduler 的 invalid-block 恢复仍是单组导向,无法安全重建递归状态,请求级 HMA 恢复留待独立改动;仅支持一个可传输 attention group。

## 关联

KV Connector 区域:#28309(KV events)、#52921(SimpleCPUOffload)。后续(正文明示):Hybrid WRITE、异构 TP 递归状态传输、请求级 HMA 恢复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
