# PR #45205: [Feat] [AMD] hicache: Enable HiCache for DSA DCP with a packed MTP draft

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @EricKing626 |
| 合并 | 2026-11-17 |
| 分类 | Speculative Decoding (tier: major) |
| Label | amd, deepseek, hicache, run-ci, run-ci-nv-extra |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 11 个文件、其中 2 个新测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/45205 |

## 一句话结论

ROCm 上为 DSA(GLM/DeepSeek)DCP 打开 HiCache(device+host L1/L2 与 L3 file/Mooncake 后端),包括把 MTP draft KV 打包进 target 的 DCP host 行;DCP 下复制的 index-K 获得独立 L3 page span(一个 L3 key 覆盖 dcp_size 个 page);GLM-5.2 AgentX agentic replay c48 的 P90 interactivity 提升 3.1x、吞吐 1.8x。

## 动机

DCP 下 DSA 的 index-K 在每个 rank 上复制,而非像 MLA KV 那样按行分片,因此需要自己的 L3 page span:一个 L3 key 应覆盖 dcp_size 个 page 而不是每 rank 一个 page。#48877 已为 row-sharded index-K 打开 DSA L3 路径,本 PR 扩展到 DeepSeek-V3.2 与 GLM 使用的复制布局,并同时放行 ROCm 上把 draft KV 打包进 target DCP host 行的 DSA MTP(EAGLE)draft,以及原本已允许的 DSPARK/DFLASH draft。

## 优化点

- 门控:`_dsa_dcp_overrides` 中 HiCache 的拒绝从"一律拒绝"改为"仅非 ROCm 拒绝"(HiSparse 仍拒绝;PD 分解除 NPU 外仍拒绝;`--speculative-eagle-topk` 非 1 仍拒绝);`resolve_hicache_dcp_compatibility` 放行 ROCm DSA MTP(`_dsa_mtp_packs_into_dcp_host` 用 `dsa_draft_shards_under_dcp` 判定 draft 是否按 DCP 分片),其他 draft 模型仍拒绝(host pool 无 DCP index translation)。
- L3 page span:`DSAIndexerPoolHost` 新增 `storage_pages_per_key` 与 `storage_page_span` 属性(复制镜像把 anchor 的一个 widened page 落成 storage_pages_per_key 页,两种布局同 span);`l3_page_span` 优先读 pool 自报 span;`_storage_page_number`/`get_data_page`/`set_from_flat_data_page`/`get_page_buffer_meta` 按 span 处理并校验对齐(ValueError)与越界(IndexError)。
- `hybrid_pool_assembler`:复制 index-K 的 sidecar host pool 页数乘 dcp_size、`storage_pages_per_key=dcp_size`;sharded 布局保持每 rank 页数。
- draft 布局检查:新增 `_check_hicache_draft_dcp_layout`(PACKED 模式要求 draft KV 与 target 一样 DCP 分片,SIDECAR 模式要求复制),不匹配即 NotImplementedError。
- `cache_controller.check_dcp_l3_extra_pool`:接受复制的 indexer mirror(pool.dcp_size==1 且 size 等于 anchor.logical_size 时,一个 key 是 dcp_size 页)。

## 关键改动

```text
python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py:
-    if cfg.enable_hisparse or cfg.enable_hierarchical_cache:
-        raise ValueError("DSA DCP does not support HiSparse or HiCache.")
+    if cfg.enable_hisparse:
+        raise ValueError("DSA DCP does not support HiSparse.")
+    if cfg.enable_hierarchical_cache and not platform.is_hip:
+        raise ValueError("DSA DCP supports HiCache only on ROCm.")
python/sglang/srt/mem_cache/hicache_storage.py:
+    span = getattr(host_pool, "storage_page_span", None)
+    if span:
+        return int(span)
python/sglang/srt/mem_cache/pool_host/dsa.py:
+        if storage_pages_per_key < 1:
+            raise ValueError("storage_pages_per_key must be at least 1")
+        self.storage_pages_per_key = storage_pages_per_key
python/sglang/srt/speculative/base_spec_worker.py: 新增 _check_hicache_draft_dcp_layout 并接入 packed/sidecar 两条 plan 分支。
test/registered/dcp/test_glm52_dcp4_hicache_amd.py(新):GLM-5.2 TP4+DCP4+MTP+HiCache 的 E2E 与强制驱逐 host reload 测试。
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。结果表:GLM-5.2-MXFP4,TP4/EP4 + DCP4,MI355X,AgentX。)

| Conc | P90 interactivity (tok/s/user) | Throughput (tok/s/GPU) | P90 TTFT (s) |
|---|---|---|---|
| 40 | 21.51 → **24.42** (+14%) | 27,749 → **29,444** (+6%) | 2.57 → **1.90** |
| 48 | 6.03 → **18.91** (3.1x) | 18,361 → **32,764** (1.8x) | 11.74 → **2.57** |

详细 agentic replay(`glm5.2-fp4-mi355x-sglang-agentic-mtp` with AgentX,aiperf,`semianalysis_cc_traces_weka_062126`,seed 42,3600 s window,HiCache arm `--hicache-size 256 --hicache-write-policy write_through`,baseline 75d7c3ba83 + PR #48773):c16 两者几乎持平(49.35 vs 49.04 tok/s/user;CPU hit 0.0%);c24 35.43 vs 35.25(CPU hit 0.1%);c40 HiCache 从 host 服务 3.4% prompt tokens,chip 命中率 96.1% → 93.4%、总命中率 96.8%;c48 host 命中 7.4%,总命中率 96.7% 对 97.1% 理论上限,无 HiCache 时被驱逐的 prefix 重新 prefill、prefill 饿死 decode。

## 验证与回归风险

单测:`test_hicache_dcp_host_pool.py` 33 pass、`test_hicache_mtp_state.py` 12 pass、`test_hicache_dsa_dcp_gate.py` 6 pass、`test_dcp_layout_unit.py` 18 pass;E2E `test_glm52_dcp4_hicache_amd.py` 9 pass(约 12.7 min);L3 file 与 Mooncake(TCP)GPU 检查通过(exact reload:六条 2,816-token prompt 重发全部 `cached_tokens_details["storage"] = 2816`;flush 后 GSM8K file 0.940/0.935、Mooncake 0.965/0.950)。E2E 精度:GSM8K 0.940(normal)/0.940(强制驱逐),阈值 0.9;KL ≤ 0.25(实测 0.07-0.15,不高于 plain TP4 + MTP 的 0.09-0.20,HiCache 无漂移)。CI 状态:Base、Extra、AMD 三项全部通过。回归风险:(1) 仅 ROCm 且 dcp_size > 1 生效,CUDA 上 DSA DCP + HiCache 仍拒绝;(2) `buffer_only` host memory 在 DCP 下仍拒绝;HiSparse、LMCache、PD 分解(NPU 除外)、eagle-topk 非 1 仍拒绝;(3) 复制 index-K 的 L3 键跨度校验(未对齐抛 ValueError、越界抛 IndexError)与 packed/sidecar 布局检查可能拒绝旧配置组合;(4) sidecar 模式沿用"多层 EAGLE 只注册第一个 draft runner"的 legacy 行为。

## 关联

#48877(row-sharded index-K 的 DSA L3 路径,本 PR 扩展其复制布局)、#48773(baseline 中修复 topk-v2 GPU illegal memory)。同批次 AMD 相关:#50656/#50660/#49955(通信)、#47052(QK-norm+RoPE kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
