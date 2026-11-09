# PR #2760: Some fixes for custom allreduce kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hanzhi713 |
| 合并 | 2024-03-22 |
| 分类 | Attention / Kernel \| tier: major |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/2760 |

## 一句话结论

修复 custom allreduce 数据竞争，改用 8 uint32 信号并加 `__threadfence_system`，杜绝卡死与乱码。

## 动机

PR 正文称近期有报告指出启用 custom all reduce 时出现 stuck generation 或 garbage text。作者在 A30、A100 上未能复现，但发现两处潜在不安全的同步：其一，使用 signal flag 时各 GPU 写同一个 8-byte signal 的不同字节，虽是 strong writes，但按 CUDA memory model 因未完全重叠不构成 morally strong，仍属 data races；其二，2-stage allreduce 或 half butterfly allreduce 需要 `__threadfence_system` 或 release-acquire 模式来绝对保证其它设备的写入在当前设备可见，而现有实现缺失。正文列出相关 issue #2788、#2742、#2731、#2728、#2795。

## 优化点

正文列出 7 项改动：1) 每个 signal 每设备改用 8 个 `uint32` 而非 8 bytes；2) 简化同步，只同步跨 GPU 同 index 的 block，保证每个 thread 的读只依赖其它设备同 id thread 的写；3) 2-stage / half butterfly allreduce 增加 `__threadfence_system` 保证可见性，正文注明这带来 a few microseconds of overhead；4) 移除对多于两个 PCIe-only GPU 的支持，因为性能提升很小；5) 增加额外 P2P 检查以避开有问题的驱动/硬件 P2P 支持；6) P2P test 增加设备数检查；7) 默认关闭 custom allreduce 一项被划掉（未采纳）。diff 中 `should_custom_ar` 对 4+ 非 NVLink GPU 直接 `return false`，`Signal` 结构由两个 8-byte union 改为按 block 索引的 `uint32` 数组，`start_sync` / `end_sync` 改为按 `blockIdx.x` 索引。

## 关键改动

```text
csrc/custom_all_reduce.cuh: 重定义 Signal 为按 block 索引的 uint32 数组，重写 start_sync/end_sync 并加入 __threadfence_system
csrc/custom_all_reduce.cu: init_custom_ar / meta_size 改用 Signal 类型，should_custom_ar 对 4+ 非 NVLink GPU 返回 false
csrc/custom_all_reduce_test.cu: 配套测试更新
vllm/model_executor/parallel_utils/custom_all_reduce.py: Python 侧 full_nvlink 改由构造参数传入，去掉 fast_cond
vllm/config.py、vllm/entrypoints/llm.py: 相关接线改动
```

```diff
 constexpr int kMaxBlocks = 64;
+// note: we don't want to use atomics for signals because peer atomics are no
+// supported on PCIe links
 struct Signal {
-  alignas(64) union {
-    uint64_t flag;
-    unsigned char data[8];
-  } start;
-  alignas(64) union {
-    uint64_t flag;
-    unsigned char data[8];
-  } end;
+  alignas(128) uint32_t start[kMaxBlocks][8];
+  alignas(128) uint32_t end[kMaxBlocks][8];
 };
```

## 性能声明

PR 正文未给出定量性能数据，仅给出定性描述：`__threadfence_system` "adds a few microseconds of overhead"；移除多于两个 PCIe-only GPU 支持的原因是 "performance improvement is small"（被删的旧注释为 4 PCIe GPUs 用 2 stage allreduce，仅在 size <= 512k 时快于 NCCL）。正文未给出 benchmark 数字，模型、硬件、精度与 workload 条件均 PR 未注明测试条件。

## 验证与回归风险

PR 正文未注明具体测试命令或结果，仅列出 Related issues：#2788（vllm 0.2.7 → 0.3.0 升级后 garbage output）、#2742（garbage output，`disable_custom_all_reduce=True` 可解决）、#2731（有人报告 `disable_custom_all_reduce=True` 解决 generation hanging）、#2728（P2P 支持）、#2795（P2P test 设备数）。diff 含 `csrc/custom_all_reduce_test.cu`。回归风险：>2 张 PCIe-only GPU 不再走 custom allreduce 而回退 NCCL；新增 `__threadfence_system` 带来微秒级开销；信号布局与同步逻辑重写影响所有启用 custom allreduce 的多卡路径，PR 未注明更多受影响路径。

## 关联

正文明确列出相关 issue：#2788、#2742、#2731、#2728、#2795（均为 issue 编号，非 PR 编号）；未给出明确的关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
