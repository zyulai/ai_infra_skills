# PR #49003: [Feat] [AMD] custom-all-reduce: Run carv2 across gfx1250 UALink nodes

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2026-11-09 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | amd, run-ci, jit-kernel, run-mori-pd, run-ci-nv-extra |
| Diff | 素材未提供 diffstat(diff 摘录完整可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/49003 |

## 一句话结论

carv2 custom all-reduce 首次跨 gfx1250 UALink 双机节点:TP16 all-reduce 较 RCCL 快 12-71%,双机吞吐 +13-31%。

## 动机

#49001 让 carv2 跑上了单节点 MI450(gfx1250),但多节点组始终用不上:ROCm 上多节点走 NVML clique 查询恒为 no,gfx1250 配置表也止步 8 ranks,跨主机 TP 组只能退回 RCCL——尽管 HIP FABRIC handle 本可在同一 UALink/UALoE fabric clique(VPOD)的主机之间映射,跨主机 TP 组理论上可复用同一套 push kernel。

## 优化点

三层机制改动,无 kernel 改动。(1) 探测:新增集体 `probe_fabric_clique`,各 rank 从 sysfs 读自己 accelerator 的 UALink 记录并 all_gather,要求 gfx1250、UALink/UALoE 且 active、PPOD UUID/VPOD ID/VPOD size 齐全且 accelerator 在 VPOD active 列表;全组共享 PPOD/VPOD、去重 accelerator 数不超过 VPOD size(DPX 分区计一次),任一字段缺失 fail closed。(2) workspace:`HipSymmetricBuffer(require_fabric=True)` 强制 FABRIC handle(POSIX fd 不能跨主机);import 失败时各 rank 先 all_gather 对齐结果再统一报错,`parallel_state` 整体回退 RCCL;`close(group)` 集体释放,先释放所有 import 再释放本机 allocation,barrier 60 s 超时则把 workspace 留到进程退出。(3) gfx1250 表新增 TP16 行(two DPX hosts):`1shot_push` 到 256 KB、`2shot_push` 到 64 MB、push grid 96 blocks(每 CU 一个),RCCL fallback 由 8 MB 推到 64 MB,push slot 增至 8 MB(TP16 每 rank 256 MB FABRIC workspace);CUDA 路径不动。

## 关键改动

```text
python/sglang/srt/distributed/device_communicators/gfx1250_fabric.py: 新文件,UALink/UALoE fabric clique 集体探测(sysfs + FABRIC 属性,失败原因逐条记录)
python/sglang/srt/utils/hip_vmm_utils.py: HipSymmetricBuffer 支持 require_fabric、teardown_barrier(60 s)与先 import 后本机的集体释放
python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py: ROCm 多节点组改走 fabric probe;多节点 close() 先 drain 再释放
python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py: gfx1250 表新增 TP16 行
python/sglang/srt/distributed/parallel_state.py + managers/scheduler.py: graceful 退出时 close_custom_all_reduce() 先关多节点 carv2
```

```python
tp16 = heuristic._replace(
    one_shot_push_threshold=256 * KB, two_shot_push_threshold=64 * MB
)
configs[16] = default._replace(
    graph=tp16, eager=tp16, num_push_blocks=min(num_sm, 96)
)
```

## 性能声明

All-reduce 延迟(TP16 跨两台 DPX 主机,bf16,us/次,max over ranks,median of 3,CUDA-graph mode,RCCL 2.32.3 fabric DDA 路径):4 KB 22.4→8.4(+62%)、64 KB 26.7→9.3(+65%)、1 MB 111.0→32.7(+71%)、64 MB 518.4→430.5(+17%),4 KB-64 MB 全区间 +12% 到 +71%;叠加单 rank ~20 us 迟到后,1 MB 111.3→46.7(+58%)。端到端:DeepSeek-V4-Flash 双机 TP8(4+4),bench_serving random-ids 1024/1024,C4-C64,输出吞吐 229.4→259.9 tok/s(C4,+13.3%)至 1,715.3→2,251.3 tok/s(C64,+31.2%),median TPOT 降 11.3-21.1%,median TTFT 降 26.5-52.0%(C64: 4,121→1,976 ms)。精度:GSM8K(1319 题,5-shot,双机 TP8)RCCL 0.948,carv2 0.945。

## 验证与回归风险

新增 CPU 单测 test_gfx1250_fabric.py(sysfs 解析、clique 校验、probe 一致性);test_custom_all_reduce_v2_capability.py 增加"ROCm 多节点只走 fabric probe、绝不触 NVML 查询"用例,并 pin 住 CUDA 拓扑路径。真机(两台 MI455X,HIP 7.16):capability 25 通过;TP2/8/16 × bf16/fp16/fp32、16 B-16 MB(TP16 到 64 MB)、eager+100 次 graph replay 0 失败;每次运行后双机 kernel log 0 fabric 事件。回归风险:ROCm graceful shutdown 改为 abort 进程组并先 close 多节点 carv2 communicator,关闭时序变化;FABRIC 释放含 60 s barrier 超时路径(超时留到进程退出);world size 9-15 仍不支持。

## 关联

#49001(前序:单节点 MI450 carv2 与 TP8 行);#48554(NVLink 多节点 import 失败统一回退的失效模式,思路在 ROCm 侧复用)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
