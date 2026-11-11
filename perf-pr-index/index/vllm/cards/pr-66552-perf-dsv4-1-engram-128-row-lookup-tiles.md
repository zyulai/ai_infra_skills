# PR #66552: [Perf][DSv4.1] Engram: 128-row lookup tiles; split GPU tables across engines

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-11-11 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, ready, deepseek |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数可见 6 个文件,另有 2 个文件因体积被丢弃,diff 末尾截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/66552 |

## 一句话结论

DeepSeek-V4.1 Engram 两项优化:大型 lookup 改 128 行 tile(GB300 prefill 每步快 12–33%、消除 50–70% 步时抖动);GPU 表可跨引擎按 NVLink 域分片,免掉每引擎 188.8 GiB 的 host 拷贝。

## 动机

Engram 大型 lookup 受读延迟限制:小 tile(16 行)下每个 program 在途读太少,`use_thp=true` 的 host 表查找使 GB300 prefill 步时在样本间摆动 50–70%(`_engram_lookup_kernel` 单次 forward 达 74 ms)。同时 GPU 表此前只能在一个引擎的 DP 组内分片,拆分部署(如 PD 分离的多个 prefill 引擎)中每个引擎都要为 CPU-offload 保留一份完整 host 表——DeepSeek-V4.1-Flash 两张表合计 188.8 GiB。对称内存(symmetric memory)无法跨引擎:引擎间无共享 process group,其 rendezvous 拒绝同节点上重复的 device ordinal,而每个引擎的 rank 都从 0 开始。

## 优化点

改动一(lookup tile):行数填满 persistent grid 的 128 行 tile 时(阈值 `rows >= 128 * max_grid`),`block_r` 从 16 提到 128、`num_warps` 4→8,每 program 在途读 8 倍;packed 行按 8 字节(WORD=tl.uint64)拷贝替代逐字节,并把 values 与 scales 两个 load 都先于 store 发出——packed lookup 若先 store 再 load 可能别名等待整个 round trip。行数不足时保持 16 行 tile;读写的行集合不变,输出 bitwise 一致。改动二(跨引擎 GPU 表分片):新配置 `peer_group_size`/`peer_group_store`,同 NVLink 域的多引擎在启动时经 torch `FileStore` 汇合一次(原子计数器定 rank、按构建顺序编号交换、各 GPU 发布 FABRIC handle),每 GPU 用 cuMem 分配 1/`peer_group_size` 行切片并映射其他 GPU 的切片(`share_peer_table`);lookup 经既有 peer-read 路径(#62091)直接从 owner 的 HBM 读,任何引擎的 step 都不等别人。配套:`process_weights_after_loading` 加载后 barrier 等全组(修了 in-engine peer 表仅靠 timing 通过的测试缺口)、表卸载时 unmap 并在所有 GPU unmap 后释放 HBM、不支持权重 reload、重启单引擎需重启全组。

## 关键改动

```text
vllm/models/deepseek_v41/common/engram.py: lookup tile 选择(128 行/8 warps);EngramPeerGroup FileStore 汇合
vllm/models/deepseek_v41/nvidia/ops/engram_fabric_table.py: (自 engram_host_table.py 重命名) share_peer_table FABRIC 分片
vllm/config/engram.py: peer_group_size/peer_group_store 配置与校验(TP=1/PP=1/DSv4.1)
docs/features/engram.md: 跨引擎分片用法文档
tests/{kernels/test_engram.py,distributed/test_engram_dp_shard.py}: tile 阈值与 peer group 测试
```

```python
max_grid = self._num_sms // 2 if background else self._num_sms
# Host reads are latency bound: once the rows fill the grid with
# 128-row tiles, each program keeps 8x the reads in flight.
block_r = 128 if rows >= 128 * max_grid else 16
...
grid = min(triton.cdiv(rows, block_r), max_grid)
_engram_lookup_kernel[(grid,)](...,
    BLOCK_R=block_r,
    WORD=tl.uint64 if wide else tl.uint8,
    num_warps=8 if block_r == 128 else 4,
)
```

## 性能声明

`vllm bench forward`,DeepSeek-V4.1-Flash(dummy weights),DEP2 单 GB300 节点,`FLASHMLA_MEGA_ATTN_DSV41` + `deep_gemm_mega_moe` + `deep_select` indexer,DSpark k=5,`--max-num-batched-tokens 34816`:use_thp=true 下每步中位时间 395.0→323.1 ms(`4*8192@131072`,33,280 tokens;样本摆动 54.6%→0.21%)、95.4→83.9 ms(`8192@65536`)、134.8→90.3 ms(`2*4096@98304`,摆动 67.2%→0.09%),即 12–33% 提速;use_thp=false 时步时不变(≤0.07% 内)。增益全部来自 lookup:kernel 每 forward 74.0→3.0 ms / 12.4→0.78 ms / 16.9→0.77 ms,其余 kernel 位移 ≤0.3 ms。`ParallelEngramEmbedding.lookup` 单独测(GB300,32M 行 FP8 表,12 heads,CUPTI+CUDA graph+cold L2):HBM 16384 tokens 前景 117.7→68.0 µs,65536 tokens 461.2→242.5 µs;pinned host unsorted 前景持平但背景抖动收敛(1517.1→1197.7 µs)。packed 行(94.4 GiB 表,32k tokens):main 263/933 µs(HBM/4 卡分片)→ 8 字节读 120/198 µs。分片显存:peer_group_size=8 时每 GPU 表占 23.60 GiB、KV cache 74.49 GiB(4→8 卡实测每 GPU 多得 23.57 GiB KV,期望 23.60);18 卡时表 10.49 GiB。GSM8K(lm-eval 5-shot,1319 题)各引擎 0.9719–0.9757,与参照一致;lookups bit-exact。

## 验证与回归风险

GB300 上 `tests/kernels/test_engram.py` 116 passed, 5 skipped;`test_engram_lookup_matches_torch` 新增 1024-token case(命中 128 行 tile 阈值),HBM 与 pinned host、packed 与 BF16 均与 torch 参考逐位一致;JIT variant 复用测试按两侧阈值扩展。`tests/distributed/test_engram_dp_shard.py` 148 passed, 4 skipped;新增 `_check_peer_group_table`(4 GPU 经 file store 分片、bit-exact 查找、释放后 HBM 归还)与 peer group 单元测试(竞争 join 下 rank 唯一、复用 store 拒绝、缺 peer 超时报 rank、reload 拒绝)。限制与风险:仅 DeepSeek V4.1(Qwen4Exp TODO)、TP=1、PP=1、表须在 HBM、需 CUDA FABRIC handle(IMEX 通道);peer 切片在 torch 外分配故 "Model loading took" 不含其量而 KV cache 估值含;任一 peer 存活期间持有死引擎的切片映射,单引擎重启即需全组重启;分片验证中 GSM8K 有两轮是在该系统自有 vLLM 构建移植 file-store rendezvous 后跑的(本分支注意力调用与该机器 FlashMLA 构建不匹配),本分支自身在 GB300 上另有一轮 4 引擎并行结果 0.9719–0.9742。

## 关联

同区域:#64609(closed,曾仅对 sorted host lookup 用 128 行 tile,本 PR 改为按行数键控、覆盖 unsorted THP 与 HBM lookup)、#64082(draft,经持久 owner 进程共享 host 表;本 PR 改为 GPU 表点对点分片、无额外进程)、#62091(既有 peer-read 路径,lookup 读他卡行的机制基础)。同模型族:同批次 PR #64715(DSv4.1 UMBP standalone)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
