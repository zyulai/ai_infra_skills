# PR #2760: Some fixes for custom allreduce kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#2760](https://github.com/vllm-project/vllm/pull/2760) |
| 作者 | hanzhi713 |
| 合并日期 | 2024-03-22 |
| labels | (none) |
| 热度 | 👍 3 · 💬 25 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
自定义 allreduce 信号同步存在 CUDA 内存模型意义下的数据竞争与缺失跨设备可见性保证,重构信号布局与同步逻辑并加 threadfence,修复生成卡死/乱码。

## 问题现象
社区报告启用 custom all reduce 后出现生成卡死(stuck generation)或输出乱码(garbage text)。作者在 A30、A100 上未能复现问题,但排查发现两处潜在不安全的同步逻辑。关联报告:#2788(vllm 0.2.7 → 0.3.0 升级后乱码)、#2742(乱码,设 `disable_custom_all_reduce=True` 解决)、#2731(一人报告 `disable_custom_all_reduce=True` 解决生成挂死)。

## 根因
正文披露两处根因:
1. 使用 signal flag 时,各 GPU 写同一 8 字节信号的不同字节。这些写按 CUDA 内存模型虽是 strong write,但因不完全重叠、不构成 morally strong,仍被视为数据竞争。
2. 2-stage allreduce 或 half butterfly allreduce 缺少 `__threadfence_system` 或 release-acquire 模式,无法绝对保证其他设备写入在当前设备可见。

## 修复方案
按正文列出的 7 项改动(diff 为 csrc/custom_all_reduce.cu/.cuh/.cu 测试与 Python 侧):
1. Signal 结构改为每设备每信号 8 个 uint32(`uint32_t start[kMaxBlocks][8]`),不再多 GPU 共写同一 8 字节;`Metadata` 结构删除,直接用 `Signal`。
2. 简化同步:只跨 GPU 同步相同 index 的 block,保证每线程的读只依赖其他设备同 id 线程的写;`start_sync`/`end_sync` 重写为按 `blockIdx.x` 索引的标志位自旋。
3. 2-stage / half butterfly 路径加 `__threadfence_system()` 保证其他设备写入可见(正文注明增加几微秒开销);2-stage allgather 改为两阶段匹配 tid 以保证同 tid 线程间可见性。
4. 移除超过两块 PCIe-only GPU 的支持(性能收益小),`should_custom_ar` 对非 NVLink 多卡直接返回 false。
5. 增加 `_can_actually_p2p` 实际往返校验,规避有 bug 的驱动/硬件 P2P 支持(可能修 #2728)。
6. P2P 测试增加设备数检查(应修 #2795)。
7. 正文原拟默认禁用 custom allreduce(已划掉);diff 显示 `vllm/config.py` 删除了此前"因稳定性问题临时强制禁用"的兜底,`LLM()` 默认 `disable_custom_all_reduce=True`,用户可显式设 False 启用。

## 影响与回归风险
- 回归面:CUDA custom allreduce 全部路径(1-stage、2-stage、half butterfly kernel、Python 侧 P2P 检测与初始化);>2 PCIe-only GPU 的 custom allreduce 能力被移除(回退 NCCL);half butterfly kernel 代码被删除,world_size==2 且非 full_nvlink 时走 1-stage。信号布局变化使 meta 缓冲区大小/偏移改变(`meta_size()` 返回 `sizeof(Signal)`)。涉及大文件传输的测试上限从 32MB 降到 8MB(diff 所示)。

## 关联
- #2788
- #2742
- #2731
- #2728(可能修复)
- #2795(应修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
