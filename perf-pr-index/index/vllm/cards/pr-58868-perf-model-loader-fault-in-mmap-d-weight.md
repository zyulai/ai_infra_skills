# PR #58868: [Perf][Model Loader] Fault in mmap'd weight pages before host-to-device copies on integrated GPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Willian-Zhang |
| 合并 | 2026-11-03 |
| 分类 | MoE(tier: major) |
| Label | ready |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(8 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/58868 |

## 一句话结论

集成 GPU(DGX Spark/GB10)上在 host-to-device 权重拷贝前先按页 touch safetensors mmap 视图,避免 page fault 落在驱动拷贝内:Qwen3.8-Flash-Next-NVFP4 主权重加载从 452-661 s 降到 25-71 s。

## 动机

对应 #58726:集成 GPU 上共享 weight loader 用 `param.copy_(loaded_weight)` 直接从 safetensors mmap 视图拷贝;对连续、同 dtype 的源,PyTorch 把 pageable 指针直接交给驱动拷贝,page fault 全部发生在拷贝内部。GB10 上每 800 KiB tensor 这样拷要 1.7-3.0 ms,而先在 CPU 侧 fault 页面只要 0.08-0.64 ms。`nvidia/Qwen3.8-Flash-Next-NVFP4` 有至少 14 万个此类 tensor,主权重加载需 452 s(NFS)/475 s(本地 NVMe)。

## 优化点

`vllm/model_executor/utils.py` 新增 `copy_weight_(dst, src)`:在集成 GPU 上对 `src` 每页 touch 一个字节(`flat[:: mmap.PAGESIZE].sum()` 加尾页)后执行 `dst.copy_(src)`;其他平台退化为普通 copy_。门控为 `current_platform.is_integrated_gpu()`(按 device 缓存);仅对连续、同 dtype、元素数 > 1 的 CPU 源 touch——非连续与 dtype 转换拷贝 PyTorch 本就经 CPU temporary(`copy_requires_temporaries`),已在 CPU 侧 fault。应用于共享 loader 的 26 个 copy 点(linear.py 8、parameter.py 6、routed_experts.py 8、vocab_parallel_embedding.py 2、default_weight_loader 2)。选择在拷贝点而非 `safetensors_weights_iterator` 做 prefault 的理由:TP 下每 rank 只 touch 自己拷贝的切片(iterator 级 prefault 在 DeepSeek-V4.1 TP4 下每 rank 要读约 287 GiB 且更慢);touch 紧贴拷贝,缓冲权重流的消费者无法在两者之间逐出页面;从 mapping 直接 serve 的张量(#58439)不拷贝即不 touch。离散 GPU 上默认关闭(Ada Lovelace 微基准显示 plain copy 在所有 cache 状态下最快,touch 反加 0.022-0.028 ms/tensor;关闭时额外检查约 0.46 µs/次调用)。

## 关键改动

```text
vllm/model_executor/utils.py: 新增 copy_weight_(dst, src) 与 _is_integrated_gpu 缓存门控
vllm/model_executor/layers/{linear.py, vocab_parallel_embedding.py, fused_moe/routed_experts.py}, parameter.py, model_loader/weight_utils.py: 26 个 copy 点换用 copy_weight_
tests/model_executor/test_utils.py: 门控强制开启下的逐字节拷贝与 mmap 页映射测试
```

## 性能声明

Boot A/B(DGX Spark GB10,TP=1,9 次完整启动交错、每次清 page cache,Qwen3.8-Flash-Next-NVFP4):主权重加载 stock 451.8 s(NFS)→ 本 PR 25.9/27.0/28.0 s;NVMe 475.2 s → 46.4/54.5 s;start→/health 584 s → 124-129 s(NFS);计数器显示 140,000 次拷贝中 139,965 次(63.4 GiB)走了 touch 路径。多来源独立复测:warm cache Qwen3-1.7B `Loading weights took` 20.2-23.7 s → 0.50/0.69 s;cold cache Qwen3.8-27B-NVFP4 主模型 96.5/96.7 s → 9.5/9.8 s(Model loading 122→16 s,start→/health 595/591→486 s),GSM8K-256 244/256 与 stock 一致;main nightly 120 GB NVFP4/FP8 主加载 504.9/534.0 s → 65.9/64.9 s(叠加 #58720 expert-name index 后 41.6/41.9 s);current head Qwen3.8-Flash-Next NVFP4(103 GiB,299,817 tensors)main 660.7 s → 71.3 s(routed experts 435→52 s,dense 63→5 s,start→/health 969→345 s)。TP=4(@gitbisector,DeepSeek-V4.1-Flash,4x GB10):`Model loading took`(最慢 rank)192-195 s → 156-161 s(-18%,read_ahead_kb=1024)、205-207 s → 149-151 s(-27%,128)。离散 RTX PRO 6000 上 touch 无差异(约 0.1 ms/tensor)。

## 验证与回归风险

新增 `test_copy_weight_prefault_copies_exact_bytes`(门控强制开启,0-d/empty/bf16/fp8/narrow/strided 源逐字节相等;CI 无集成 GPU,是该路径唯一覆盖)与 `test_copy_weight_prefault_maps_the_mmap_view`(/proc/self/smaps 验证调用后整个视图已映射);CPU 可运行 loader 测试 216 passed / 5 skipped(38 个失败为容器无 GPU,改动前后相同);9 次启动的 greedy 输出与 first-token logprobs 完全一致(touch 只读源,权重逐字节相同)。风险:Jetson Orin/Thor 等其他集成设备会走 touch 路径但未测试;GH200 的 `is_integrated` 返回值未检查;model-specific 自行 `.copy_()` 的 loader 不经 helper(Qwen4Exp PLE shard copy 为 follow-up);mmap 加载在该 128 GB 机器上会推 3.2-4.4 GiB 其他内存进 swap(stock 同样如此,且 vLLM 把 swapped 内存计入 KV cache sizing)。

## 关联

#58726(动机 issue);#58720(expert-name index,叠加后主加载 41.6/41.9 s);#49991(ROCm iterator 内 clone)、#50794(Ascend H2D 前 clone)为其他平台对同一拷贝点的不同动因处理;#57973/#55985 为替代加载路径;#58857/#58782(移除 DeepSeek-V4 VL wrapper 的 sorted 缓冲)与本 PR 互补。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
