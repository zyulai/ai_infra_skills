# PR #64994: [Perf][DCP] Up-project the MLA decode output before the A2A merge

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @whx-sjtu |
| 合并 | 2026-11-07 |
| 分类 | Speculative Decoding(tier: major) |
| Label | ready, ci/build, kimi, k3 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64994 |

## 一句话结论

DCP A2A merge 前先做 W_UV up-projection,每 head 传输 514→130 bf16 值,C32 decode 吞吐 +1.1%。

## 动机

`--dcp-comm-backend a2a` 下,MLA decode 先 all-to-all 每个 rank 的 partial latent output,LSE merge 之后才 apply `W_UV`;all-to-all 每 head 携带 514 个 bf16 值,而 up-projection 后只需 130(Kimi-K3,512/128)。LSE merge 是 per-head 加权和、`W_UV` 每 head 线性,先投影在数学上只差 rounding。

## 优化点

`MLADCPManager.v_up_proj_combine` 仅为不带 PCP 的 NCCL all-to-all 设置;symmetric-memory 与 `ag_rs` 路径不变。`MLAAttention` 与 Kimi-K3 `MultiHeadLatentAttention`(含 DSpark draft 的)在 `W_UV` 未量化时启用,默认开启;两种顺序都 round 两次到 activation dtype。代价:gathered `W_UV`(Kimi-K3 TP8/DCP8 下 +12 MiB,合 0.32 GiB per GPU;local `W_UV` 是 `kv_b_proj` 的 view,不增加)与一个对 N 倍 heads 的 BMM;低于 8 个 decode tokens 时该 BMM 成本超过 all-to-all 的节省,故先 merge——阈值 `_V_UP_PROJ_FIRST_MIN_TOKENS = 8`(MI355X DCP8 实测)。限制:未在 CUDA 上运行、未从本分支端到端;DeepSeek-V3.2 attention 与 direct symmetric-memory A2A 保持 merge-then-project。

## 关键改动

```text
vllm/v1/attention/ops/dcp.py: 新增 dcp_a2a_v_up_proj_lse_reduce():先 bmm 到 V 维再 A2A,低于阈值回退 merge-then-project;MLADCPManager 挂 v_up_proj_combine
vllm/model_executor/layers/attention/mla_attention.py: all-gather W_UV_dcp;decode 合并走 v_up_proj_combine 时跳过 _v_up_proj
vllm/models/kimi_k3/nvidia/mla.py: 同上,Kimi-K3 MLA 接入
tests/distributed/test_dcp_a2a.py: 新增 2-rank 分布式测试(-k v_up_proj)
```

```python
# Below this many tokens, reading every head's W_UV costs more than the smaller
# All-to-All saves (measured on MI355X, DCP8).
_V_UP_PROJ_FIRST_MIN_TOKENS = 8
```

## 性能声明

- 微基准(8x MI355X,96 或 64 heads,cold weights):project-then-merge 在 1–4 tokens 为 +1.1 µs(+4–5%),8 tokens 亦然;16–360 tokens 快 12% 到 48%。
- Payload per rank per layer @160 tokens:15.8 MB→4.0 MB(computed)。
- 端到端(非本分支:runtime patch,nightly `3966924d`;Kimi-K3 1P1D prefill/decode 分离,2 节点 8x MI355X,TP8+DCP8 per role,`--dcp-comm-backend a2a` 且 `VLLM_USE_DIRECT_DCP_A2A=0`,FP8 KV,DSpark 4 draft tokens,Mooncake RDMA,agentic multi-turn trace):decode-bound C16/C32:throughput +0.6%/+1.1%(before-merge 5,257.4 vs 对照 5,227.0;10,492.2 vs 10,379.9 tok/s/GPU),P90 interactivity +1.4%/+2.8%;decode step −1.1%/−1.7% at 24/32 running requests;prefill-bound C72:interactivity +4%,throughput flat。GSM8K:C32 0.958(两序相同)、C72 0.963。
- 数值检查:8-rank 仿真(96 gathered heads,12/rank,L=512,V=128,bf16,含 NaN 空 shard),两序对 fp64 参考的相对误差同为 2.30e-3(32 requests)。

## 验证与回归风险

`pytest tests/distributed/test_dcp_a2a.py -k v_up_proj`:4 passed(错误 head 顺序会 fail);该测试注册进 `.buildkite/test-amd.yaml` 与 `.buildkite/test_areas/distributed.yaml` 的 Distributed Features CI。`MLADCPManager` 的三个 selection 测试断言 `v_up_proj_combine` 的有无。风险:仅影响 a2a、无 PCP、`W_UV` 未量化的 MLA decode 路径;8-token 阈值只在 MI355X 测量;既有两个测试文件各有 7 个 main 上也存在的失败(fp32 LSE 对 bf16 参考,CUDA-only direct ops),非本 PR 引入。

## 关联

#48196(draft,冲突:把同一投影放进更大的 sparse-MLA 改动,先于 `MLADCPManager`,覆盖所有 DCP backend)。无其他。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
