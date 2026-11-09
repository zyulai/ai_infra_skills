# PR #65214: [HiSparse] Support the extensible KV cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-11-03 |
| 分类 | Attention / Kernel \| tier: major |
| Label | nvidia, mrv2, kv-cache-manager, hisparse |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/65214 |

## 一句话结论

HiSparse 接入 extensible KV cache，默认配置 GPU+host KV 容量 +16%，吞吐与 main 持平。

## 动机

extensible KV cache（#56492）此前把 HiSparse 排除在外。本 PR 让 HiSparse 能跑在 extensible KV cache 上：其 device pool 像其他 KV cache 一样被预留与增长，而 host pool 仍留在 extensible 体系之外。移除该 gate 还暴露出两处 warmup 崩溃，一并修复。

## 优化点

host_resident tensors 不进入 extensible 路径：`ExtensibleKVCache`、`committed_views` / `committed_kv_cache_tensors` 与 `shrink_kv_cache_configs` 都跳过 host_resident 张量。extend 后的 cache dict 与固定大小路径保持一致：runner 保留 `init_kv_cache` 返回的 `kv_caches_by_layer`，`extend_kv_cache` 换入 committed views 并传给 runner.kv_caches 与各 connector，`rebind_hisparse_device_caches` 把 resident / hot views 重新绑定到 committed 前缀；由于基地址不变，extend 前捕获的 CUDA graphs 仍然有效。另修两处 warmup 崩溃：HiSparse autotune 在调优前先 commit `cdiv(topk, hot_block_size)` 个 blocks；context-prefill warmup 把 reservations clamp 到 spec 的 block-table 行长度（`max_num_blocks_per_req`），Mamba 在非 align 模式下豁免。

## 关键改动

```text
vllm/v1/worker/extensible_kv_cache.py: extensible KV cache 跳过 host_resident tensors、维护 committed views
vllm/v1/core/kv_cache_planning.py: KV cache 规划与 warmup reservation clamp
vllm/v1/worker/gpu/model_runner.py: 保留 init_kv_cache 字典并把 committed views 传给 runner 与 connectors
vllm/v1/worker/gpu/warmup.py: 修复 HiSparse autotune 与 context-prefill warmup 崩溃
vllm/v1/hisparse/binding.py: HiSparse resident / hot views 的重新绑定
```

## 性能声明

正文声明默认 flags 下 "GPU+host KV capacity grows by 16% and throughput matches `main`"，GSM8K 在重度 host offload 下也与 main 持平。表格（B200、`VLLM_USE_V2_MODEL_RUNNER=1`、base 3b6eeb0606；GLM-5.3-NVFP4、TP4、fp8 KV cache、block-size 64、random 16384 输入 / 1024 输出、host_pool_gib 64，每次重启 server、跑两次）：GPU+host KV tokens 由 1,313,935（main）增至 1,529,536（PR），日志显示 `Extended KV cache to 367616 blocks (63.11 GiB)`；c=16 吞吐 661/666 → 668/664 tok/s、TPOT 20.3/20.1 → 20.1/20.2 ms；c=64 吞吐 949/944 → 956/962 tok/s、TPOT 54.3/54.7 → 53.9/53.5 ms。GSM8K（1319 题、5-shot、并发 32、bfloat16 KV、host_pool_gib 32）：PR 0.904 / 0.912，main（--num-gpu-blocks-override 6208）0.923 / 0.914。

## 验证与回归风险

全部通过的测试：test_hisparse.py 6/6、test_extensible_kv_cache.py 20/20、test_kv_cache_planning.py 66/66、test_gpu_warmup_blocks.py 36/36、test_mooncake_store_worker.py 248/248、test_nixl_connector_hma.py 98/98；另有一例 NIXL HMA 因需无法下载的 gated checkpoint 未跑。新增 test_hisparse_extensible_kv_cache（greedy 输出与 post-extend `kv_caches_by_layer` 须等于同 block 数的固定 cache，CI 在 H200 运行，故只有 SM100 覆盖 autotune 修复）与 test_shrink_kv_cache_configs_leaves_host_resident_tensors。P/D 验证：NIXL 与 Mooncake store 的 MultiConnector 均能启动服务，greedy 输出与无 P/D 的 HiSparse 一致（5/5 prompts、48/48 tokens）。风险：NIXL/Mooncake 注册本身没有常驻测试（无 HiSparse P/D CI job）；sleep/wake 与 HiSparse 在 extensible cache 上的组合未覆盖。

## 关联

正文提及 extensible KV cache 来自 PR #56492。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
