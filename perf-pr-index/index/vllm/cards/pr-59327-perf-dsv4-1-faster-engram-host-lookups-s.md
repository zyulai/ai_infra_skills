# PR #59327: [Perf][DSv4.1] Faster Engram host lookups: sorted rows, inline big lookups

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-10-01 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | DSv4.1, deepseek, ready |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 4 个文件,3 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/59327 |

## 一句话结论

DeepSeek-V4.1 Engram CPU offload 表查找按表偏移排序行以共享 TLB 页遍历,大 lookup 改为主流内联,GB200 上内核最高 3.73×、E2E TTFT 最高 1.23×。

## 动机

DSv4.1 的 Engram n-gram 表从 CPU offload 内存(pinned 或 /dev/shm)查找。在 GB200 上瓶颈是 GPU 地址翻译而非带宽:表远大于 TLB 覆盖,每行都在 backing 小页上 miss TLB,各行独立付出页遍历代价。此外大 lookup 跑在 prefetch 流上时,主流上的持久化 kernel(MegaMoE、mHC、block-scaled GEMM)无法与 lookup CTA 共存,后台 lookup 反而拖慢主流,EP collective 又要等最慢 rank。另有两处顺手修复:/dev/shm 空间检查误用总大小(应查剩余空间),int64 id 因输出行索引绑定 id dtype 而无法编译。

## 优化点

两个机制改动:(1) 排序查找——超过行数阈值时,host 表 lookup 先把行按表偏移(`ids - vocab_start` 展平后 `sort()`)排序再 gather,输出写回用 `sorted_dst` 索引,使相邻行共享页遍历;THP(huge page)表跳过,低于阈值的 lookup 也跳过;(2) 内联大 lookup——token 数(max over DP ranks)超过阈值时,lookup 从 prefetch 流改到主流上以全宽 grid 执行。阈值由 `_engram_lookup_thresholds` 按 GPU 家族给定:Blackwell(GB200 实测)为 36864 行 / 4096 tokens,其他 GPU(含 ROCm)两条路径均关闭、行为与之前一致。同时把全部 Engram 逻辑合并进 `common/engram.py`(删除 `nvidia/engram.py`),NVIDIA 与 AMD 共享一份实现;新增 DP 共享 /dev/shm 存储(`DPSharedEngramStorage`)与 huge-page 分配/`MADV_COLLAPSE` 路径。

## 关键改动

```text
vllm/models/deepseek_v41/common/engram.py: SORTED 查找内核变体、
  _engram_lookup_thresholds、内联/后台分派、DPSharedEngramStorage
vllm/models/deepseek_v41/nvidia/engram.py: 删除(并入 common)
vllm/models/deepseek_v41/amd/model.py: 改从 common 导入 Engram
tests/kernels/test_engram.py: 排序查找重建 checkpoint、offload 构造等测试更新
```

关键机制摘录(engram.py,按表偏移排序):

```python
local = indices[:, self.head_start : head_end] - self.vocab_start_idx
indices, dst = local.flatten().sort()
dst = dst.int()
```

## 性能声明

内核微基准(单 GB200 绑 NUMA 节点,DSv4.1-Flash 表 24 heads × 16M rows / 94.4 GiB,均匀随机 id,CUPTI 中位数,16 组 id 轮换;HEAD 与新实现输出 bitwise 一致,64 tokens 以上新内核在排序阈值下最多慢 1.5%):

- DEP2/DEP4(/dev/shm 表 DP 共享):32768 tokens 时 36963.3 → 9905.1 us(3.73×),16384 时 2.26×,8192 时 1.56×,4096 时 1.22×;64-1024 tokens 约 1.00-1.02×;
- TP2/TP4(pinned 表):TP4 8192 tokens 877 → 160 us,TP2 32768 tokens 14319 → 4291 us。

E2E prefill(DeepSeek-V4.1-Flash,DEP4 = DP4×TP1+EP,单 GB200 节点,DeepGEMM mega MoE + FlashMLA mega attention + fp8 KV,`--max-num-batched-tokens 16384`,关 prefix caching,单 token 输出):concurrency 64 时 TTFT p50 4k 1059→863 ms(1.23×)、32k 8620→7056 ms(1.22×),吞吐 222,676→270,319 tok/s;concurrency 1 时 16k/32k TTFT 1.09×。作者估计 64 并发下每步约省 50 ms(排序约 20 ms + 内联约 30 ms)。注:concurrency 4/16 的 TTFT 受到达顺序影响、重复波动最高 40%,故未列入。

## 验证与回归风险

测试:`tests/kernels/test_engram.py` 与 `tests/distributed/test_engram_dp_shard.py`,4×GB200 上 117 passed,ruff/format/mypy 通过。新覆盖:排序 lookup 在 TP padding 下仍能重建 checkpoint;/dev/shm 空间不足或查询失败时回退分片且 leader 广播决策。风险点:排序/内联仅 Blackwell 启用(GB200 实测值),其他家族待调参;内核输出与 HEAD bitwise 一致但吞吐路径变化(内联与主流争 SM)在非测试配置下未验证;依赖 `dp_metadata` 与 Engram DP 组元数据。

## 关联

- 相邻 PR:#58766(FP8 rows into wkv,改同一 lookup 内核)、#58310(表大于 host 内存时的检查)、#57195(SP rank 共享 host 表)——均未做排序或内联,但触及同一批文件
- 同区域:DSv4.1/DSv4 系列 Engram 与 MoE 优化

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
