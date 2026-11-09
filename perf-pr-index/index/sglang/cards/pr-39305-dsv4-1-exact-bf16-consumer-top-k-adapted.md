# PR #39305: [dsv4.1] Exact bf16 consumer top-k adapted from DeepSelect

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-09-13 |
| 分类 | Sampler / Detokenization(major) |
| Label | jit-kernel |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 3 个文件(JIT kernel 头文件、Python op 封装、correctness 测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/39305 |

## 一句话结论

把 DeepSeek-V4.1 两级 sparse indexer consumer 步的近似 bf16 top-k kernel 换成源自 DeepSelect 的精确选择,B200 上最高提速约 22%(rows=512/valid=4096:10.83→7.36us)。

## 动机

consumer 步要在每行最多 16384 个 bf16 sparse logits 中选 top-512 并经 per-row 物理块表做 page transform,原 kernel 是占位实现:单一 13-bit fp16-derived histogram 在选择边界上是近似的,1024 线程 × 16 元素的配置也未调优(代码里留有 "not optimized at all" 的 TODO)。索引质量与延迟都受影响。

## 优化点

取 DeepSelect init-window 选择算法(32 KiB 窗口全部驻留寄存器的 top-k),去掉 TMA/streaming 外壳,以纯 SIMT 重写并针对 16384 宽、k=512 特化。机制:两轮 radix(先对 raw bf16 高字节做 256-bin histogram,再对与 pivot 同高字节的元素做低字节 histogram)精确定位第 k 大值;全程使用 "distorted" 位序(负数翻转并排在正数之下);打包 hit 测试用 `__hgt2_mask`/`__heq2_mask` + `prmt` + `dp4a` 把每元素的命中压成 1 bit,每 warp 一次前缀扫描加一次 shared atomic 得到块级 (gt, eq) 前缀,再用 equal quota 保证恰好取 k 个。本移植特有的优化:每行一个 512 线程 CTA,每线程 32 个分数(4×16B 向量)全程驻寄存器,40 寄存器/线程、3 CTAs/SM、无 spill;负半区高字节 histogram 偏移 16 词避免 +x/-x 同指数的 smem bank 冲突(实测占一半 atomic wavefronts);行尾部分向量用负 NaN padding(在 distorted 序中低于 -inf,永不入选);-0 pivot 特判;smem 与 histogram 联合复用共 9 KiB,输出一次 coalesced page-transformed 写出。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh: kernel 重写,近似 13-bit histogram → 精确两轮 radix,kMaxTopK 4096→2048,kBlockSize 1024→512、occupancy 2→3
python/sglang/kernels/ops/attention/dsv4/topk.py: topk_transform_bf16_small 文档更新(精确选择、k≤2048、NaN 不支持)
test/registered/kernels/ops/attention/test_topk_bf16.py: 新增奇数宽度/重度 tie/signed zero/denormal/NaN 等 100 cases
```

## 性能声明

PR 注明条件:B200、`marker.do_bench`(CUDA graph、L2-rotated inputs)、k=512、含 page transform、随机 bf16 分数、permuted block table,旧 kernel → 本 PR(单位 us):

- valid_len=16384:rows=1 5.24→4.93;rows=32 5.51→5.18;rows=128 7.09→6.58;rows=512 15.94→12.38。
- valid_len=9000:rows=512 13.03→9.03。valid_len=4096:rows=512 10.83→7.36。
- 超过约 450 行后 GPU 饱和(3 CTAs/SM),时间随行数线性:2048 行 36us、8192 行 134us。
- k=2048 非目标形状:小 batch 下比旧 kernel 慢 10-15%(epilogue 只用 512 线程),512 行时快 10%。

## 验证与回归风险

`test_topk_bf16.py` 扩展到 100 cases:所有 case 以 multiset 对比 `torch.topk` 并用逆表验证 page transform,覆盖奇数行宽(部分尾向量)、重度 tie/常量/全 -inf 行、pivot 附近的 signed zero、denormal 与全范围 bf16 位型、k 上下的正/负 NaN。`test/registered/attention/unittests/dsv4/test_dsv41_sparse_indexer.py` 通过。风险:k 上限从 4096 收窄到 2048(超出会 RuntimeCheck 失败);NaN 分数现在明确不支持(正 NaN 会占用槽位,输出为 k-n 个真实分数 + -1);CI Base/Extra/AMD ROCm 运行在本 PR 快照上均为失败状态。

## 关联

上游算法来源 DeepSelect(deepseek-ai/DeepSelect)。同区域为 DeepSeek-V4.1 sparse indexer kernel 系列,本卡未收录其他 PR,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
