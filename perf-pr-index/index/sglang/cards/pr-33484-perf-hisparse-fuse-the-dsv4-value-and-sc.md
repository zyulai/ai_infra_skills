# PR #33484: perf(hisparse): fuse the DSv4 value and scale swap-in copy on ROCm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AMD-yanfeiwang |
| 合并 | 2026-08-10 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/33484 |

## 一句话结论
ROCm 上把 DSv4 稀疏 KV swap-in 的 value+scale 两次拷贝融合为一次 73-word 拷贝,swap-in 延迟最高降约 20%。

## 动机
DeepSeek-V4 的 C4 token 为 584 字节但非连续:page row 里是 64 个 576B value 加 64 个 8B scale。ROCm 的 swap-in 对一个 token 要调两次 `transfer_item_warp`(576B 一次、8B 一次),在 wave64 上是三个 wavefront pass(64 value words、8 value words、1 scale word),只有第一个 pass 能让整个 wavefront 忙碌。`item_size_bytes` 是运行时参数,编译器保留带 `s_waitcnt` 的 rolled loop,三个 pass 串行等待主机内存往返;而 `swap_in_selected_pages` 每 decode step 在主 stream 上按层运行于 indexer 与 attention 之间,处于 decode 关键路径。CUDA 分支的 `device::hisparse::transfer_item` 本就一起搬 value 尾部与 scale,只是 warp 逻辑硬编码 32 lanes 无法在 ROCm 使用。

## 优化点
新增 `transfer_dsv4_item_warp`:把一个 token 当作单个 73-word 空间遍历(576B value 72 words + 8B scale 1 word),value 尾部与 scale 落在同一 pass;pass 数是编译期常量,循环可完全 unroll,所有 load 先于任何 store 发出,把暴露的主机内存往返从 3 次降到 1 次。仅替换 ROCm `IsDsv4Layout` miss copy 处对两次 `transfer_item_warp` 的调用,不改 API/config/模板;CUDA 行为与通用(非 DSv4)路径均不动。PR 还记录了一个负结果:128-bit 单 pass 变体虽 pass 更少,但只有 37/64 lane 参与,在所有 batch size 上都更慢(batch 1 时反而比基线慢),故采用 64-bit two-pass 形态。

## 关键改动

```text
python/sglang/kernels/jit/csrc/hisparse.cuh: 新增 transfer_dsv4_item_warp(单 73-word 空间、staged load-then-store);ROCm DSv4 miss copy 改用它
test/registered/kernels/ops/kvcache/test_hisparse.py: 新增 test_load_cache_to_device_buffer_dsv4_fused_copy_multi_miss
```

## 性能声明
MI355X (gfx950),DSv4 paged-layout swap-in,`miss_rate=0.2`、`hot_buffer_size=4096`、`top_k=2048`;两变体编入同一 kernel 运行时选择、同一进程内交替测量,15 reps × 5 timed rounds 取中位数、两次独立运行(batch 1:102.4/105.7us → 87.8/87.9us,-14.2%/-16.9%;batch 10:-20.9%/-19.8%;batch 24:-8.5%/-8.0%;batch 100:-3.6%/-3.4%)。高 batch 下增益收窄,因为 kernel 变为带宽受限。另测得 miss_rate 0.2→0.001 时 batch 1 从 110.7us 降到 20.5us,即 82% 的 swap-in 时间是 miss copy。

## 验证与回归风险
新增 `test_load_cache_to_device_buffer_dsv4_fused_copy_multi_miss`:单次 launch 内多个 miss、源/目的 page offset 各异(含第二页),按内核输出取目的地并校验未写 slot 保持填充;作者称砍掉 scale 半边拷贝会同时挂掉新测试与既有 `dsv4_mla_miss_copy_layout`。MI355X 全文件 10 passed、2 skipped(CUDA-only DSv4 paged 用例)。风险面小:改动限于 ROCm `#ifdef` 分支,不触碰 CUDA 与非 DSv4 路径。

## 关联
#33085(拓宽通用路径,分支不相交、无重叠)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
