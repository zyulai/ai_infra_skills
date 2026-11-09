# PR #28609: Facade DSA index-cache: MTP topk-reuse state + index-K storage

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-08-06 |
| 分类 | Speculative Decoding |
| Label | deepseek, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用(4 个文件从 diff 中截断) |
| 链接 | https://github.com/sgl-project/sglang/pull/28609 |

## 一句话结论

为 DSA index-cache 引入门面(Facade)封装:统一 MTP top-k 复用状态与 index-K 存储的所有权,消除裸写 ForwardBatch/spec_info 的脆弱生命周期。

## 动机

DSA index-cache 逻辑此前没有单一 owner:MTP top-k 复用状态散落在对 `ForwardBatch`/`spec_info` 的裸状态写入上,生命周期脆弱(依赖逐处手工维护 `reuse_dsa_topk_indices`、seed capture 等);index-K 的分配、读写、move、传输元数据与 CPU offload 则与 latent-KV pool 逻辑交织。layer-skip 策略已由 `dsa_layer_skips_topk` 拥有,本 PR 有意不涉及。

## 优化点

新增两个门面类并重接线调用方:`IndexTopKShareState` 是 `reuse_dsa_topk_indices` 与 `spec_info.dsa_topk_indices` 之上的适配器,提供 `mtp_iteration` 上下文管理器——进入时按需保留 step 0 的 draft-extend seed,退出时(含异常路径)总是清空复用状态;`publish` 统一负责把 topk_indices 写回 spec_info 或 seed capture 缓冲。`IndexKeyCache` 拥有 DSA index-K 缓冲,收拢 read/store/move/transfer-metadata/CPU offload/load 操作,`DSATokenToKVPool` 经 `_create_index_key_cache` 委托并保留 `index_k_with_scale_buffer` 兼容属性;`LayerSplitIndexKeyCache` 服务上游 DSA layer-split pool,保留 owned-only 分配、owner-broadcast 读取、remote buffer 失效、fused-read 时 latent-KV 预取顺序、PD state-buffer 元数据与 HiCache CPU 拷贝语义。NextN decoder 与 EAGLE V2 draft loop 改经状态适配器。缓冲布局与 `index_buf_accessor` codec 不变,分页大小/量化布局/传输计数行为保持。

## 关键改动

```text
python/sglang/srt/layers/attention/index_topk_share.py: 新增 IndexTopKShareState 与 mtp_iteration
python/sglang/srt/mem_cache/index_key_cache.py: 新增 IndexKeyCache(index-K 缓冲门面)
python/sglang/srt/mem_cache/dsa_cache_layer_split.py: LayerSplitIndexKeyCache 替换内联逻辑
python/sglang/srt/mem_cache/memory_pool.py: DSATokenToKVPool 委托到 IndexKeyCache
python/sglang/srt/models/deepseek_nextn.py / deepseek_v2.py: 经适配器读写 topk 复用状态
```

```python
@classmethod
@contextmanager
def mtp_iteration(cls, forward_batch, enabled=True, keep_carry_seed=False):
    if not enabled:
        yield None
        return
    spec_info = forward_batch.spec_info
    forward_batch.reuse_dsa_topk_indices = True
    if not (keep_carry_seed and spec_info.dsa_topk_indices is not None):
        spec_info.dsa_topk_indices = None
    try:
        yield cls.from_mtp_carry(forward_batch)
    finally:
        spec_info.dsa_topk_indices = None
        forward_batch.reuse_dsa_topk_indices = False
```

## 性能声明

PR 未给出性能数字,属结构/健壮性重构;行为兼容性是主要声明:索引缓冲布局与 accessor codec 不变,layer-split rank 只为 owned 层分配完整缓冲(与改前一致),非 layer-split 池保持每层一个完整缓冲,DSV4 nested 与 NPU 池维持原实现。

## 验证与回归风险

验证:改动文件的 pre-commit/py_compile/冲突标记检查通过;聚焦测试(状态、shard、pool、HiCache/offload、DSA EAGLE CUDA-graph)53 passed 1 skipped;GB300 上两 rank `test_dsa_layer_split_broadcast.py` 通过;既有 real-DSA 验证覆盖 plain decode、HiCache offload/load 与 PD 分离的 index-buffer 传输;固定种子 EAGLE3 baseline vs candidate 在 40/40 个 GSM8K 样本上生成文本、finish_reason、解析答案、分数、stop/error rate 全部一致,合并后官方 Llama checkpoint 复跑再次 40/40 一致(97.5% stop rate,0% 错误)。风险面:DSA/EAGLE/MTP 全路径的生命周期语义集中到门面后,任何依赖旧裸写行为的旁路代码(如自定义 pool 子类)需对齐;pack 显示 CI Base 待跑、Extra 通过。

## 关联

同区域:同批 #32196(PD decode 下 EAGLE DP 图与 token 元数据一致性,同样围绕 EAGLE 状态生命周期);`dsa_layer_skips_topk` 的 layer-skip 策略为既有独立 owner,不在本 PR 范围。索引中另有 DSA/NSA 元数据优化卡(如 pr-14781、pr-17554)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
