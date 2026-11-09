# PR #29163: [Core] Add xxHash as a high-performance hash option for accelerating prefix caching

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LuminolT |
| 合并 | 2025-12-03 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, llama, qwen, deepseek, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `vllm/utils/hashing.py`、`vllm/config/cache.py`、两个新增 benchmark 脚本与测试/文档 |
| 链接 | https://github.com/vllm-project/vllm/pull/29163 |

## 一句话结论

为 prefix caching 增加可选 xxHash(XXH3-128)哈希,微基准比 SHA-256 快约 1.5x,块级哈希吞吐 +25-40%。

## 动机

#23673 为 prefix caching 引入 sha256 与 sha256_cbor,但 prefix caching 不需要密码学安全性:不存储秘密数据、不跨进程边界、只用于标识前缀块的结构等价。SHA-256 因此是纯开销;Python 内建 `hash()` 又因加盐且跨进程不稳定不可用。xxHash 在两者之间取平衡:碰撞抗性远好于内建 hash,速度远快于 SHA-256。

## 优化点

以可选依赖方式接入:`vllm/utils/hashing.py` 尝试 import `xxhash` 包(要求提供 `xxh3_128_digest`,否则视为不可用),新增 `xxhash`(pickle 序列化 + XXH3-128)与 `xxhash_cbor`(canonical CBOR 序列化 + XXH3-128),并注册进 `get_hash_fn_by_name`;`CacheConfig.prefix_caching_hash_algo` 的 Literal 扩为四种,默认仍为 sha256,文档明确写非密码学哈希在多租户下的碰撞/泄露风险由用户权衡。`kv_cache_utils` 的 `PYTHONHASHSEED` 未设置警告从仅 sha256_cbor 扩展到所有 CBOR 系函数(新增 `_CBOR_HASH_FUNCTIONS` 集合)。另附两个可移除的基准脚本(微基准 + 块级哈希吞吐)与 CLI 文档。

## 关键改动

```text
vllm/utils/hashing.py: xxhash/xxhash_cbor 函数(XXH3-128)与注册
vllm/config/cache.py: PrefixCachingHashAlgo 扩展,默认仍 sha256
vllm/v1/core/kv_cache_utils.py: _CBOR_HASH_FUNCTIONS 覆盖 PYTHONHASHSEED 警告
benchmarks/benchmark_hash.py 与 benchmark_prefix_block_hash.py: 新增基准
tests/v1/engine/test_engine_args.py: xxhash CLI 参数测试
```

关键片段:

```python
def xxhash(input: Any) -> bytes:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return _xxhash_digest(input_bytes)  # xxh3_128_digest
```

## 性能声明

微基准((32-byte bytes, 32-int tuple),10000 次):SHA256(pickle)1.45±0.24 μs,xxHash(pickle)0.99±0.16 μs,内建 hash() 0.19 μs;非内建中 xxHash 比 SHA-256 快约 1.5x。块级哈希吞吐(20k blocks × 64 tokens):sha256 29.14M tokens/s → xxhash 36.51M tokens/s(xxhash 提升 25-40%;cbor 变体 6.58M → 6.97M tokens/s)。端到端 `benchmark_prefix_caching.py`(单卡 A100,OPT-125M,input 128-256,output 128,200 prompts × repeat 5):sha256 约 4.17 s → xxhash 约 4.12 s,提升 1-2%(小模型、低命中率场景)。长跑吞吐与 GC(input 1500/output 1:219.9 → 223.4 req/s,+1.6%;output 150:77.9 → 78.9 req/s,+1.2%):GC 总时长 decode-heavy 场景 1.92 s → 1.68 s(-12.8%),无新增长尾;作者预期生产环境命中率更高时收益更大。

## 验证与回归风险

验证:CLI 参数测试(未安装 xxhash 时 skip)、两个基准脚本、端到端 prefix caching 基准与长跑 GC 对比。风险:引入第三方依赖 `xxhash`(刻意保持 optional,严格安全环境可不装);非密码学哈希理论上有碰撞风险,配置文档明确警示多租户场景需自行评估;CBOR 变体的可复现性依赖 PYTHONHASHSEED;默认算法未变,不 opt-in 无行为差异。

## 关联

前序:#23673(sha256/sha256_cbor 引入)。同分类 KV Cache/Memory:#60084、#64094。GC 分析复用 Jialin/vllm#4 的方法(与 #23673 相同)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
