# PR #31926: [Mooncake] Fix silent SSD offload corruption when TP/PP ranks share ssd_offload_path

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31926](https://github.com/sgl-project/sglang/pull/31926) |
| 作者 | ironashram |
| 合并日期 | 2026-09-17 |
| labels | hicache, run-ci, bypass-fastfail |
| 热度 | 👍 4 · 💬 13 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论

各 TP/PP rank 的 Mooncake client 共享同一 ssd_offload_path,bucket 文件名碰撞后静默互相截断、读到错误 KV;改为每 rank 一个私有子目录,并补 dp_rank 消歧。

## 问题现象

每个 scheduler rank 内嵌自己的 `MooncakeDistributedStore` client,而 `MooncakeStore.__init__` 给所有 client 传入同一个 `ssd_offload_path`。Mooncake 的 SSD tier 假设一个存储位置只有一个 client,共享目录会静默损坏(silent corruption):碰撞发生时 bucket 文件被截断,后续读取返回错误的 KV 字节,且**没有任何报错**。正文给出生产部署实例:8 个 client 在同一秒启动、共享同一个种子,几分钟内即互相截断 bucket。影响面为启用 Mooncake SSD offload 的多 rank(TP/PP/DP)部署。

## 根因

正文披露三点机制(mooncake-store/src/storage_backend.cpp):

1. bucket id 按进程生成——以 `timestamp<<12` 为种子加普通 atomic increment,同一秒启动的进程种子相同;
2. 重启后每个 client 都从相同的 scanned max_bucket_id 恢复,使文件名碰撞是确定性的;
3. bucket 文件以 `O_CREAT|O_TRUNC` 打开,碰撞即截断另一个 rank 的 bucket。

另有一个加重因素:dp-attention 下 `HiCacheController` 上报的 tp_rank 是 attention-group-local rank,当 attn_tp_size == 1 时(如 tp8/dp8 的 DeepSeek)每个 DP rank 都是 0,仅靠 tp_rank 无法区分,所有 rank 会塌缩到同一目录。

## 修复方案

为每个 rank 分配私有子目录:`ssd_offload_path/rank_<dp_rank>_<tp_rank>_<pp_rank>_<attn_cp_rank>`,在 `setup()` 之前 `os.makedirs(..., exist_ok=True)` 创建;后缀沿用该文件已有的 per-rank key 约定(mha_suffix)。`HiCacheStorageConfig` 新增 `dp_rank` 字段(默认 0),由 controller 已计算的 dp_rank 填充;`storage_config` 为 None 时路径原样使用,行为不变。

关键摘录(mooncake_store.py):

```diff
-                    setup_kwargs["ssd_offload_path"] = self.config.ssd_offload_path
+                    ssd_offload_path = self.config.ssd_offload_path
+                    if storage_config is not None:
+                        ssd_offload_path = os.path.join(
+                            ssd_offload_path,
+                            f"rank_{storage_config.dp_rank}"
+                            f"_{storage_config.tp_rank}_{storage_config.pp_rank}"
+                            f"_{storage_config.attn_cp_rank}",
+                        )
+                    os.makedirs(ssd_offload_path, exist_ok=True)
+                    setup_kwargs["ssd_offload_path"] = ssd_offload_path
```

改动文件:`mooncake_store.py`、`hicache_storage.py`、`cache_controller.py`,并新增单测 `test_mooncake_ssd_offload_path.py`(覆盖 tp/pp/dp/cp 各维度目录唯一性)。模型输出无变化(仅客户端存储路径处理)。

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围,不写。
- 回归面:按 diff,仅 Mooncake SSD offload 的路径选择逻辑与 HiCacheStorageConfig 字段;不涉及 KV 写读语义。正文明确迁移注意:旧布局下已 offload 到目录根部的 bucket 被本次 layout 改动 orphan——一次性缓存丢失(miss 时重算,不是损坏),这些旧文件可删除。storage_config 为 None 的调用路径保持原行为。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
