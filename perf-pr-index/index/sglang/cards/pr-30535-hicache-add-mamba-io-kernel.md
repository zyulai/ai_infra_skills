# PR #30535: [hicache]: add mamba_io_kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jojoakm |
| 合并 | 2026-07-16 |
| 分类 | Attention / Kernel |
| Label | run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 4 个文件(3 新增 1 修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/30535 |

## 一句话结论

为 HiCache 的 Mamba 状态传输新增 JIT CUDA kernel 后端,低并发下比 MLA kernel 快 12-14 倍。

## 动机

SGLang HiCache 已支持 Mamba 模型(如 Qwen-Next),但其状态在 host/device 间的 KV cache 传输此前只支持 `direct` I/O backend(纯 DMA)。DMA 在低并发、小批量 item 场景下无法有效利用 PCIe 带宽。本 PR 引入专用 JIT 编译的 CUDA kernel 作为 `kernel` I/O backend,采用 cache-bypass load/store 与软件流水来提高 PCIe 传输效率;block-cooperative 设计在低并发下相对现有 MLA kernel 最高加速 14 倍。

## 优化点

新增两个方向相反的传输 kernel:`transfer_mamba_load_kernel`(page_first → layer_first,逐层 load)与 `transfer_mamba_backup_kernel`(layer_first → page_first,全层 backup)。每个 block 固定 1024 线程,一次处理 16 字节(uint4),使用 `load_nc`/`store_nc` 绕过 L2 缓存,并用 double buffering(先发下一步的 load 再 store 上一步的结果)维持尽可能多的 outstanding PCIe 事务。关键在于 block-cooperative:多个 block 通过 grid-stride 协作处理同一 item,使单个 item 也能用满 32 个 warp;而 MLA kernel 是 flat 的 warp-to-item 映射,item 少时大部分 warp 空闲,这就是低并发 12-14x 差距的来源。集成侧:`MambaPoolHost` layout 限制从仅 `page_first_direct` 放宽到 `page_first`/`page_first_direct`,移除 load/backup 路径的 `io_backend='direct'` 强制,`io_backend='kernel'` 时路由到新 kernel,并将 `can_use_write_back_jit` 置 True 以配合 HostPoolGroup 的 AND 逻辑(MLA staged kernel 需要索引留在 CPU,Mamba 路径自行把索引搬上 GPU)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/kvcacheio/transfer_mamba.cuh: CUDA kernel 实现(load/backup 两 kernel)
python/sglang/jit_kernel/transfer_mamba.py: JIT wrapper,暴露 transfer_kv_mamba_pf_lf / transfer_kv_mamba_lf_pf
python/sglang/srt/mem_cache/memory_pool_host.py: MambaPoolHost 接入 kernel 后端,放宽 layout/io_backend 限制
test/registered/jit/test_transfer_mamba.py: 新增单元测试
```

```cuda
for (int64_t item_id = blockIdx.x; item_id < params.num_items; item_id += gridDim.x) {
  uint4 v_cur = device::details::load_nc(...);           // cache-bypass load
  for (off = base + kBytesPerBlockPerStep; off < item_size; off += kBytesPerBlockPerStep) {
    uint4 v_next = device::details::load_nc(...);        // 预取下一步
    device::details::store_nc(..., v_cur);               // store 上一步
    v_cur = v_next;
  }
}
```

## 性能声明

- Micro-benchmark(page_first src,H2D,mamba kernel vs MLA kernel):4 items / 512KB 时 0.53ms vs 6.47ms(12.3x,39.7 vs 3.2 GB/s);4 items / 1MB 14.0x;4 items / 2MB 14.5x。低并发(4 items)约 43 GB/s vs 3 GB/s;高并发(64+ items)两者均打满 PCIe(约 41-44 GB/s),差距消失(16/64/256 items 下约 0.9-3.7x)
- E2E(H20,tp=4,Qwen-Next,input=4096,output=128):kernel vs direct 基本持平——conc 1:TTFT 219 vs 213ms;conc 16:1288 vs 1215ms;conc 256:14727 vs 14843ms;吞吐 1.21/5.41/7.51 vs 1.20/5.84/7.49 req/s。作者解释:prefill 计算主导 TTFT,稀释了传输优势
- E2E 未声称有整体收益;收益集中在 host-device 传输这一段

## 验证与回归风险

新增 `test/registered/jit/test_transfer_mamba.py`:backup/load roundtrip、empty/single/full indices、fp16/bf16、两种 layout,并验证非目标位置不被误写;已注册 CUDA/AMD CI。精度测试:Qwen-Next tp=4,`--hicache-write-policy write_through --hicache-io-backend kernel`,多并发(1-256)输出与 direct 后端一致。风险:item_size 强制要求 16 字节对齐;kernel 要求索引张量在 CUDA 上(路径中做 `to(cuda)`);`can_use_write_back_jit` 强制 True 与 MLA staged kernel 的 CPU 索引约定存在耦合,改 HostPoolGroup 逻辑时需注意。

## 关联

同区域:HiCache 既有 MHA/MLA staged write-back JIT kernel(共享 `sglang.jit_kernel` 的 `load_jit` + `cache_once` 机制,具体 PR 编号素材未注明)。被替换路径:`transfer_kv_per_layer_mla_pf_lf` 等 MLA kernel 在 Mamba pool 的调用。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
