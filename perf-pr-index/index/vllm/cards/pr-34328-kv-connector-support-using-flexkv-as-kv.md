# PR #34328: [KV Connector] Support using FlexKV as KV Cache Offloading option.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @feiqiangs |
| 合并 | 2026-03-12 |
| 分类 | KV Cache / Memory |
| Label | documentation, ready, v1, kv-connector, nvidia |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/34328 |

## 一句话结论

新增 FlexKVConnectorV1,把 KV cache 卸载到腾讯 FlexKV 分布式多级缓存:ISL=21K/OSL=1K/bs=8 下 TTFT -60%、QPM +16%。

## 动机

大规模 LLM 推理下,GPU 显存内的 prefix cache 容量有限,长前缀场景重复 prefill 开销大。FlexKV 是腾讯云 TACO 团队与社区协作开发的分布式 KV store 与多级缓存管理系统,靠多级缓存(CPU 内存、SSD、远端存储)与分布式 KVCache pool 让推理引擎获得更高吞吐、更低延迟;vLLM 此前没有接入它的 KV Connector。

## 优化点

实现 `FlexKVConnectorV1`(`KVConnectorBase_V1` 子类),本体是薄代理层:所有公共方法委托给 `flexkv.integration.vllm.vllm_v1_adapter.FlexKVConnectorV1Impl`,flexkv 未安装时抛出带安装指引的 ImportError。核心设计是 KV 传输全部在 scheduler 侧管理:`build_connector_meta` 负责发起传输任务、`update_connector_output` 轮询完成状态,KV block 直接在 FlexKV server 与 vLLM GPU 显存间传输(与 NIXL 类似);worker 侧 `start_load_kv`/`save_kv_layer`/`wait_for_save` 等 hook 目前为 no-op,仅为满足接口契约并预留 layer-pipelining 扩展点。`request_finished` 在请求结束、block 释放前触发异步 offload,async save 期间 block 不释放,直到 `get_finished` 返回对应 request id。经 `KVConnectorFactory` 注册后,以 `--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'` 启用。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py: 新文件,connector 代理层
vllm/distributed/kv_transfer/kv_connector/factory.py: 注册 FlexKVConnectorV1
examples/offline_inference/prefix_caching_flexkv.py: 对比 baseline / prefix caching / FlexKV 的示例
tests/v1/kv_connector/unit/test_flexkv_connector.py: mock flexkv 的单元测试
docs/features/disagg_prefill.md: 用法文档
```

```python
try:
    from flexkv.integration.vllm.vllm_v1_adapter import FlexKVConnectorV1Impl
except ImportError as e:
    raise ImportError(
        "FlexKV is not installed. Please install it to use "
        "FlexKVConnectorV1. See https://github.com/taco-project/FlexKV "
        "for installation instructions."
    ) from e
self._flexkv_connector = FlexKVConnectorV1Impl(vllm_config, role)
```

## 性能声明

- ISL=21K, OSL=1K, batch_size=8:TTFT 降低 60%,QPM(queries per minute)提升 16%;TPOT 增加 13%(即 decode 有额外开销)。
- PR 未注明测试条件(未给出模型/硬件/精度/workload 细节)。

## 验证与回归风险

`tests/v1/kv_connector/unit/test_flexkv_connector.py` 以 mock 的 flexkv 包运行:验证缺安装时的 ImportError 信息,以及 shutdown、start_load_kv、get_num_new_matched_tokens、request_finished 等全部公共 API 到 impl 的逐一委托;示例脚本另验证三种模式的生成文本一致。风险:实际 KV 传输逻辑在 FlexKV 仓库,本 PR 仅覆盖代理层;运行依赖外部安装 flexkv 与 `FLEXKV_CONFIG_PATH` 配置;worker-side hook 全为 no-op,vLLM connector 接口演进时需同步维护;性能数字来自 PR 作者自有场景,未附可复现脚本。

## 关联

同为 KV offloading/transfer 的既有选项:OffloadingConnector、NixlConnector、MooncakeConnector(见 diff 文档与 factory 上下文)。索引同区域另有 pr-42296(DSv4 in SimpleKVConnector)、pr-60695(mooncake EAGLE attention 修复)、pr-63827(defer async KV loads)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
