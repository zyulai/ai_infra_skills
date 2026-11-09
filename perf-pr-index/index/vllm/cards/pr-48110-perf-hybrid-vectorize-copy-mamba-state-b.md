# PR #48110: [Perf][Hybrid] Vectorize _copy_mamba_state_block to uint64 for temporal

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fuscof-ibm |
| 合并 | 2026-07-18 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际仅改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/48110 |

## 一句话结论

Mamba/hybrid 模型 temporal state 拷贝从 1 字节改为 uint64 向量化访存:低请求数 kernel 提速 5.9-6.5x,GB200 MTP E2E 吞吐最高 +2.97%。

## 动机

PR #40172 引入 `postprocess_mamba_fused_kernel`,在 hybrid 模型开启 prefix caching(align 模式,含 MTP)时拷贝状态;#42406 把拷贝逻辑重构进 `_copy_mamba_state_block`(MRv2)。每次被接受的 draft step 跨 block 边界都会触发全量 `(num_reqs x total_states)` 拷贝——Qwen3.5-9B 为 24 个 linear-attention 层 x (conv+temporal) = 每 step 48 次拷贝,每 block 约 2 MiB temporal + 80 KiB conv;temporal 状态是 conv 的 20-30 倍,主导拷贝量。原拷贝体使用 1 字节 load/store,无法利用硬件带宽:H100 上 kernel 峰值仅约 62% HBM3,GB200 上约 33% HBM3e。

## 优化点

将 temporal state 拷贝改为 uint64 load/store(每指令 8B 事务,LDG.64/STG.64):bf16/fp16/fp32 的连续 inner dim 本身 8 字节对齐,src/dst 地址均按整 token slice(inner_size * elem_size)对齐,copy_size 是其倍数,因此主体循环以 uint64 步进,再用最多 7 字节的 masked byte tail 处理尾部(仅 sub-8B slice 可达)。conv 路径保持原 1 字节循环不动。为使每个发出的地址"由构造即安全",`MambaSpecDecodeGPUContext` 初始化时新增 assert:temporal state 基指针与 per-block stride 都必须 8 字节对齐——assert 作为廉价护栏而非 fallback。kernel 由此转为 HBM 带宽受限。

## 关键改动

```text
vllm/v1/worker/mamba_utils.py: _copy_mamba_state_block 的 temporal 分支改为 uint64 向量化主循环 + byte tail;initialize_from_forward_context 增加 8B 对齐 assert
```

关键代码摘录(节选,<=15 行):

```python
copy_size_u64 = copy_size // 8
src_u64 = src_addr.to(tl.pointer_type(tl.uint64))
dst_u64 = dst_addr.to(tl.pointer_type(tl.uint64))
for i in range(0, copy_size_u64, COPY_BLOCK_SIZE):
    mask = (i + offsets) < copy_size_u64
    data = tl.load(src_u64 + i + offsets, mask=mask)
    tl.store(dst_u64 + i + offsets, data, mask=mask)
tail_start = copy_size_u64 * 8
tail_bytes = copy_size - tail_start
```

## 性能声明

微基准(生产 grid(num_reqs, 48),Qwen3.5-9B 真实状态布局,tp=1、num_spec=2,每请求 49.88 MiB 拷贝,src/dst block 分离以避免 L2 掩蔽 DRAM 流量):
- H100(HBM3 ~3.35 TB/s):reqs=1 5.90x(108.6→641.0 GB/s);reqs=8 3.47x;reqs>=16 达 HBM 峰值 81.8-84.3%(main 上限 62.3%);reqs=128 kernel 时延 6417.25→4742.66µs(1.35x,省 1674.6µs)。
- GB200(HBM3e ~8 TB/s):reqs=1 6.50x(73.8→479.4 GB/s);reqs=8 5.57x;reqs=32 达 ~73.8% 峰值(main 上限 32.6%);绝对节省随时延单调增长 1.20→2.82ms(reqs=128 5136.81→2313.11µs,2.22x)。

端到端(GB200,Qwen3.5-9B,TP=1,MTP qwen3_next_mtp num_spec=2,prefix caching,INPUT_LEN 500/OUTPUT_LEN 5330/500 prompts,每并发 4 次取后 3 次):
- output tok/s:+0.75%(conc=32)/ +2.97%(conc=64)/ +1.48%(conc=128);median TPOT 相应 -0.87% / -4.03% / -1.32%。
- MTP acceptance 不变(两分支均约 98.5-98.7%)。

## 验证与回归风险

- 验证:自建微基准 harness(bench_copy_mamba_state_block.py)与 E2E vllm bench serve A/B(main 4e5ca89c vs PR 分支);pack 未列 pytest 单测改动。
- 回归风险面:所有走 `_copy_mamba_state_block` 的 hybrid/mamba 模型(MRV1/MRV2 align 模式 prefix caching + MTP);8B 对齐由 setup assert 保证,若未来出现非 8B 对齐的 state dtype/布局会直接报错而非静默错误;conv 路径未动。新增 assert 只在 temporal state 初始化时执行,运行时无开销。

## 关联

- 前序:PR #40172(fused triton kernel for gpu freq sweep / postprocess_mamba_fused_kernel 引入)、PR #42406(copy 机制重构进 _copy_mamba_state_block)。
- 同区域:索引库已有 pr-49436(3D grid tiling of state copy)、pr-28587(mamba2 torch-compile)、pr-63742(mamba shared checkpoint)等 hybrid state 管理卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
