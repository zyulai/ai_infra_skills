# PR #64715: [ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4.1-Flash

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-11-10 |
| 分类 | Speculative Decoding (tier: major;classify 输出,内容实为 KV connector) |
| Label | documentation, rocm, deepseek, kv-connector, DSv4.1 |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数为 14 个文件(diff 完整,未见截断) |
| 链接 | https://github.com/vllm-project/vllm/pull/64715 |

## 一句话结论

让 DeepSeek-V4.1-Flash 在 ROCm 上以 standalone UMBP KV pool 运行:sparse MLA 快路径不再被 connector 关掉,TP 复制的 MLA KV 池内只存一份;工作集溢出 GPU KV 时 agentic 吞吐 +46%。

## 动机

此前 ROCm 上任何 KV connector 都会关掉 DeepSeek-V4.1 的两条快路径——`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` 与 NaN-free extra-cache——因为旧判定只看"是否配置了 kv_transfer_config",而 connector 可能写入非本 cache writer 产生的字节,aiter kernel 又不 scrub NaN。同时 standalone UMBP pool 未适配 TP>1:MLA latents、SWA、indexer K 这类 TP 复制的 KV 组会在池里存 `tp_size` 份。目标是在 MI355X agentic recipe 下让被逐出 GPU prefix cache 的前缀可从节点本地 DRAM 池恢复,而不牺牲 attention 快路径。

## 优化点

三层机制。第一,引入 `only_restores_canonical_kv()` classmethod(base 默认 `False`):UMBP 在 `key_namespace` 为 auto 时返回 `True`(派生 namespace 对每层 spec 做指纹,保证对象都是同 layout 写入的整块),`MultiConnector` 对全部子 connector 取 AND;`kv_connector_may_load_foreign_kv()` 据此替换 `deepseek_v41/amd/rocm.py` 里的 `kv_transfer_config is not None` 判定,使 UMBP 场景保留 aiter sparse MLA 与 NaN-free 路径,其他 connector 语义不变。第二,TP 复制 KV 去重:`tp_replicated_group_ids()` 找出所有层 `max_tp_shards=1` 的 cache group(TP>1 且无 context parallelism 时),standalone runtime 声明 `shares_pool_across_ranks=True`,此类组的 key 用 `tp*` 取代 TP rank——TP rank 0 写、所有 rank 读,池内一份;scheduler 按 `keys_per_object()` 逐组计算命中宽度。第三,namespace 隔离:引擎若配有"可能加载外来 KV"的兄弟 connector(`MultiConnector` 组合),其 namespace 追加 `foreign_kv` 字段,避免外来字节流入 UMBP-only 引擎还原的 canonical namespace。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/base.py: 新增 only_restores_canonical_kv() 钩子(默认 False)
vllm/distributed/kv_transfer/kv_connector/v1/umbp/data.py: tp_replicated_group_ids();BlockIdentityCodec 支持 tp* 复制组键
vllm/distributed/kv_transfer/kv_connector/v1/umbp/connector.py: UMBP 实现 canonical 判定、dedup 配置与握手元数据
vllm/distributed/kv_transfer/kv_connector/v1/umbp/worker.py: 非 rank0 跳过复制组 store(_owned_store_plans)
vllm/models/deepseek_v41/amd/rocm.py: 快路径判定改用 may_load_foreign_kv
vllm/models/deepseek_v41/attention.py: MLAAttentionSpec 声明 max_tp_shards=1
```

```python
def stores_group(self, group_id: int) -> bool:
    return self.tp_rank == 0 or group_id not in self.replicated_groups

# key_for_namespace(): 复制组用 TP 无关键
tp = "*" if group_id in self.replicated_groups else tp_rank
return (
    f"umbp:vllm:v1:{self.namespace.value}:"
    f"tp{tp}:pcp{pcp_rank}:dcp{dcp_rank}:"
    f"pp{pp_rank}:g{group_id}:{block_hash.hex()}"
)
```

## 性能声明

验证环境:单节点 2×MI355X,MORI v1.2.3.post1(含 UMBP),128 GiB standalone DRAM 池,InferenceX recipe `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`(DSpark MTP、monokernel、mxfp4 indexer)。GSM8K(lm_eval 5-shot,1319 题,num_concurrent=64):baseline strict-match 0.9212±0.0074,UMBP pass1 0.9333±0.0069,pass2(92.7% prompt token 来自池)0.9295±0.0071,差距在约 1.2 combined standard errors 内;mean acceptance length 3.43/3.42/3.44 持平。AgentX(replay `semianalysis_cc_traces_weka_062126`,DURATION=3600,总吞吐/GPU):TP2 c128 245.3K→246.8K tok/s(+0.6%,全 fits,无开销);TP2 c192 179.3K→262.5K(**+46%**,TTFT mean/p90 7.12/19.46s→1.59/4.18s,TPOT 58.0→37.0ms,computed prefill −42%);DP2 c256 270.3K→280.6K(+3.8%,TTFT −17%/−7%,TPOT −4%)。

## 验证与回归风险

新增单测集中在 `tests/v1/kv_connector/unit/test_umbp_core.py`:namespace 对 MultiConnector 兄弟的隔离、`tp_replicated_group_ids` 拓扑参数化、rank0-only store、scheduler 采纳 worker 复制组握手、`only_restores_canonical_kv` 配置矩阵。已知限制(正文):仅单节点;ROCm 无 SWA replay,SWA 组仍整体存储;extensible KV cache 走 VMM 分配,standalone server 无法经 HIP IPC 映射,当前 main 对 ROCm UMBP 自动关闭,perf 两侧均传 `--no-enable-extensible-kv-cache` 以保证 baseline 与 UMBP 臂 KV 容量一致。其他 connector、embedded UMBP、非 ROCm 平台行为不变。

## 关联

RFC 前序:#57187(UMBPStoreConnector 设计)。同模型族优化:同批次 PR #66552(DSv4.1 Engram lookup tiles / GPU 表分片)。pack 内无其他同分类 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
