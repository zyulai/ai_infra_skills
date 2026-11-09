# PR #60103: [Perf][HiSparse] Fuse per-layer slot invalidation and shorten the all-resident resolver path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-10-29 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | ready, kv-connector, hisparse |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明「2 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/60103 |

## 一句话结论

HiSparse 将 slot 失效合为每步一次 launch，B200 上该步 374–759 µs→3.5–162 µs，resolver 每层提速 1.16×–3.69×。

## 动机

HiSparse 在 decode 阶段每层都有 GPU time 与 launch 开销：slot 失效目前是 forward graph 之后每个 leader layer 一次 eager launch；resolver 还需 per-layer 的 `copy_` + `masked_fill_` 屏蔽 CUDA-graph padding 行，main 上单个 block 串行屏蔽整段 padding，在 padded batch 上主导该 launch；resolver 的整数除法也占据 per-position 指令。PR 旨在削减这些 per-layer decode 开销。

## 优化点

slot 失效改为每步一次 launch：新 kernel 在 (written tokens × index groups) 上遍历，使用缓存的各组 `device_global_indices` device 表，`HiSparseRuntime.invalidate_written_slots` 由 `HiSparseSlotInvalidator` 取代；该表按 leader set 缓存、首次使用时构建，稳态步骤无 H2D 拷贝，并校验所有 index groups 共享同一 hot layout。resolver 自行依据 `num_valid_rows` 屏蔽 CUDA-graph padding 行（每个 padding row 一个 block），去掉 per-layer `copy_` + `masked_fill_` 以及 `request_ids`/`row_indices` buffer。算术层面，2 的幂块运算改用 shift/mask（64 行块时编译期展开），并在 request lookups 完成前预取首行 top-k（`kPrefetchedTopK`）。

## 关键改动

```text
csrc/hisparse_kernels.cu: 新增 block_div/block_mod 位移掩码运算、kPrefetchedTopK 预取、template<kBlockSize> 的 hisparse_resolve_residency_kernel，resolver 自屏蔽 padding 行
vllm/v1/hisparse/runtime.py: 以 HiSparseSlotInvalidator 取代 invalidate_written_slots，缓存 device_global_indices 表，实现每步单次失效 launch
vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py: 接入新的失效与 resolver 调用路径
csrc/ops.h、csrc/torch_bindings.cpp: 暴露新 kernel 的接口与绑定
tests/v1/attention/test_sparse_mla_backends.py、tests/v1/worker/test_utils.py: 新增/更新 HiSparse 相关单测
```

## 性能声明

测试条件：B200（微基准为同进程 main vs PR kernel、CUPTI device time，输出逐点 bit-identical）；端到端为 GLM-5.3-NVFP4、8×B200 TP8、16k in / 1k out，main 与 PR 交错各 2 reps、相同 KV 容量（1.49M GPU+host tokens）。Slot 失效每步（78 = GLM-5.3 index groups）：78×8 从 374 µs（78 launches）→3.5 µs（1）、78×64 634→13.0 µs、78×256 759→59.1 µs（61×64 367 µs→9.9 µs）。Resolver 每层（top-k 2048，CUDA graph）：64 req / 0% host 9.2→4.6 µs（2.00×）、100 req MTP3 (400/512) 48.8→13.2 µs（3.69×）、128 req MTP3 / 100% host 54.7→47.2 µs（1.16×），全部 36 个扫点快 1.16×–3.69×。端到端 mean TPOT：c=16 15.84/15.67→15.47/15.46 ms（-1.8%），c=64 40.23/40.35→40.12/40.02 ms（-0.5%），no HiSparse 基线 38.27 ms；PR 注明仅弥合 c=64 时 HiSparse 相对 no-HiSparse 约 2.0 ms TPOT 差距的约 11%。

## 验证与回归风险

测试：`pytest tests/v1/attention/test_sparse_mla_backends.py -k hisparse ...` 在 B200 上 103 passed（新测 `test_hisparse_invalidation_covers_every_index_group`），`pytest tests/v1/e2e/general/test_hisparse.py` 5 passed，GLM-5.3-NVFP4 serve A/B。微基准输出与 main 逐点 bit-identical。基准对比 main `d0c20bb840` 与 PR `ab5afbdab6`；其后 main 合并的 #62592 仅将 op 调用改走 HiSparse platform backend，单测已在其上重跑。风险：约束所有 index groups 共享同一 hot layout（checked）；resolver 改动覆盖 padded batch、MTP3 与不同 host page 比例路径；PR 注明 no open PR 触及这些 kernel，剩余 c=64 约 1.8 ms TPOT 开销在本 PR kernel 之外。

## 关联

#62592（其后 main 的合并，把 op 调用改走 HiSparse platform backend，单测在其上重跑）；PR 明确注明 no open PR 触及这些 kernel。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
