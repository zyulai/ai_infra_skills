# PR #23673: [Core] Use sha256 bytes instead of BlockHash to reduce GC overhead

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @linzebing |
| 合并 | 2025-09-09 |
| 分类 | KV Cache / Memory |
| Label | documentation, performance, ready, v1 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 3 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/23673 |

## 一句话结论

用 sha256 bytes 替代 BlockHash NamedTuple,消灭每块 token tuple 的短命对象,GC 压力大降,e2e 吞吐 +5.33%。

## 动机

v1 前缀缓存的 `BlockHash` 是一个 NamedTuple,除 hash 值外还保存整块 token IDs 的 Python int tuple 与 extra keys 以防哈希碰撞;`BlockHashWithGroupId` 又在其上再包一层。每个 block 因此持有多个短命 Python 对象。Jialin 的分析:1500 token prompt、block_size 16 时每请求约 93 个 block,每 block 约 4 个可回收对象,即每请求 372 个对象;Python 默认 GC 阈值 700,意味着约每 1.88 个请求就触发一次 GC collect——过于频繁,调度路径被 GC 打断,产生吞吐抖动。

## 优化点

把 block hash 的表示整体换成 bytes:`BlockHash = NewType("BlockHash", bytes)`,`BlockHashWithGroupId` 也用 raw bytes 表示——`make_block_hash_with_group_id` 把 group_id 以 4 字节大端 append 到 hash 尾部,pair 不再创建 tuple。SHA-256 全 256 位摘要使碰撞实际不可能,token IDs 与 extra keys 不再需要嵌入 hash 结构,`hash_block_tokens` 只返回一个 bytes。配套改动:`vllm.utils.sha256` 直接返回 digest bytes(不再 `int.from_bytes`),`sha256_cbor_64bit` 更名为 `sha256_cbor` 且不再截断 64 位;移除 "builtin"(Python hash)选项,`prefix_caching_hash_algo` 默认改为 "sha256";NONE_HASH 变为 bytes;KV events 的对外 hash 表示保留 `ExternalBlockHash = Union[bytes, int]` 与新环境变量 `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`(默认 True)以向后兼容旧的 64-bit int 事件消费方。代价是 sha256 比 builtin hash 每次慢约 1µs,但换来短命对象数量大幅下降、GC 事件减少。

## 关键改动

```text
vllm/v1/core/kv_cache_utils.py: BlockHash/BlockHashWithGroupId 改为 bytes NewType,pack/unpack helper,移除 token_ids/extra_keys 字段
vllm/utils/__init__.py: sha256 返回 digest bytes,sha256_cbor 替代 sha256_cbor_64bit,移除 builtin hash
vllm/config/cache.py: PrefixCachingHashAlgo 只剩 sha256/sha256_cbor,默认 sha256
vllm/envs.py + vllm/distributed/kv_events.py: ExternalBlockHash 与 KV events int 兼容开关
vllm/v1/core/block_pool.py: cache/lookup 路径改用 make_block_hash_with_group_id 与 maybe_convert_block_hash
```

关键片段(group id 打包进 bytes key):

```python
def make_block_hash_with_group_id(block_hash: BlockHash,
                                  group_id: int) -> BlockHashWithGroupId:
    return BlockHashWithGroupId(block_hash +
                                group_id.to_bytes(4, "big", signed=False))
```

## 性能声明

H100,facebook/opt-125m,benchmark_throughput.py:

- input 1500 / output 150 / 10000 prompts:105.25 → 110.86 requests/s(+5.33% 吞吐;total tokens/s 173665.94 → 182910.25,output tokens/s 15787.94 → 16628.46),GC 事件总数与离群 GC 事件均减少(截图)。
- input 1500 / output 1 / 20000 prompts(跑 3 次):235.92 → 239.29 requests/s(+1.43%)。
- 哈希方法微基准(32-byte bytes / 32-int tuple,10000 次):SHA256 1.17µs、SHA256_CBOR_64BIT 6.37µs、builtin hash 0.18µs(builtin 比 SHA256 快 6.4x、比 CBOR 快 34.8x);即每次 hash 慢约 1µs,但 GC 节省远大于此开销。

## 验证与回归风险

测试:`tests/v1/core/test_kv_cache_utils.py`、`test_prefix_caching.py`、`test_single_type_kv_cache_manager.py` 的 hash 参数网格改为 [sha256, sha256_cbor] 并断言 bytes 语义;`tests/v1/engine/test_engine_args.py` 更新默认 algo 与 CLI 校验;`tests/utils_/test_utils.py` 校验 sha256 返回 bytes;GSM8K eval(Llama-3.1-8B-Instruct,5-shot,limit 250):main 与本 PR flexible-extract 0.792 / strict-match 0.764 完全一致。回归风险:移除 "builtin" hash 选项是接口破坏性变更;KV events 事件体默认仍转 64-bit int,消费端若直接用 bytes 需关 `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`;pickle 序列化 hash 输入的 CPU 开销在 token 生成路径上每块约 +1µs。

## 关联

同区域:v1 前缀缓存(BlockPool/KVCacheManager/kv_cache_utils)及其 GC 相关分析(引用 @Jialin 的 GC 分析分支与 hash 方法 benchmark gist);pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
