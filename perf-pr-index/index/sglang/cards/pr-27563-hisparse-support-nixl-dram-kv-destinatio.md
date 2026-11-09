# PR #27563: hisparse: support NIXL DRAM KV destinations for HiSparse

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ishandhanani |
| 合并 | 2026-06-27 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | documentation, high priority, run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/27563 |

## 一句话结论

让 NIXL PD 分离可把 KV 直传 HiSparse 的 host(DRAM)decode 池,decode 满载场景吞吐 1.82x、TTFT 约 5x 改善。

## 动机

decode 侧 HiSparse 把 KV transfer buffer 放在 host 内存,但 NIXL 路径此前把这些指针一律注册为 `VRAM`,导致 PD 分离 + HiSparse 组合根本无法工作(目标缓冲实际是 DRAM)。失败模式不是性能问题而是功能打通:要在 NIXL 注册与传输描述符中显式携带内存类型元数据,才能把 KV 直接写进 host-backed HiSparse 池。

## 优化点

核心是 NIXL 的内存类型(memory kind)管道:`KVArgs` 新增 `kv_data_mem_kinds`,与 KV 指针同长度传输;prefill 缓冲保持 `VRAM`,decode 侧 HiSparse host KV 池标 `DRAM`(普通 dense decode KV 仍 `VRAM`),draft pool 与 DSV4 的 device 部分(如 c4 层之后)保持 `VRAM`。NIXL 侧:注册时按 memory kind 分桶(`register_memory` 按 VRAM/DRAM 分别调用,DRAM 的 device_id 用 0),bootstrap ZMQ 消息携带 kinds 与 per-buffer item_lens(带向后兼容:缺失时默认全 VRAM);传输描述符按源/目标各自 kind 构建。DeepSeek V4 的混合内存 HiSparse 由 `_kv_xfer_mem_segments` 按 contiguous `(src_kind, dst_kind)` 对切分传输段,每段独立 prepped dlist,并用 `kvpart` multipart 完成通知聚合到原 chunk 语义;homo-TP 直传 host 时目标 dlist 用 decode 的 slot stride 几何 + 源的传输长度(HiSparse host pool 的 slot stride 可以与 prefill 不同)。次要改动:HiSparse swap-in JIT kernel 的 launch block size 从硬编码 1024 变为 `SparseConfig.swap_in_block_size`(默认 512,`--hisparse-config` 校验 [1,1024])——B200 复现器在 1024 线程下失败、512/768/960 下通过,默认 512 保留可行路径并留平台调优空间。

## 关键改动

```text
python/sglang/srt/disaggregation/decode.py: _init_kv_manager 按 enable_hisparse
  生成 kv_data_mem_kinds(host 池 DRAM,device/draft 部分 VRAM)
python/sglang/srt/disaggregation/nixl/conn.py: kind 归一化/打包、分桶注册、
  _kv_xfer_mem_segments 混合内存分段、send_kvcache_mixed、kvpart 通知
python/sglang/srt/managers/hisparse_coordinator.py: swap_in_block_size 配置化,
  替换硬编码 block_size=1024
```

```python
kv_data_mem_kinds = (
    ["DRAM"] * len(kv_data_ptrs)
    if self.scheduler.enable_hisparse
    else ["VRAM"] * len(kv_data_ptrs)
)
```

## 性能声明

Saturated GLM-5.1-FP8 TP8/DP8 NIXL PD 分离 benchmark(random 32000 in / 8000 out,256 prompts,max-concurrency 256,request-rate inf,disable-radix-cache,decode 侧 `--enable-hisparse` + top_k=2048/device_buffer 6144/host_to_device_ratio 10/swap_in_block_size 512):HiSparse on 输出吞吐 2470.19 tok/s、mean TTFT 120357.76 ms、mean ITL 75.72 ms、每 DP 峰值 32 并发、0 次 retraction;HiSparse off(dense)1356.67 tok/s、TTFT 604836.69 ms、ITL 31.47 ms、峰值 7 并发、56 次 KV-full retraction。即 1.82x 吞吐、约 5.0x lower mean TTFT;机制是 KV 压力下更高常驻 decode 并发(dense ITL 更低但留不住长请求)。作者注明 HiSparse-on 有 3 个 NIXL bootstrap room 传输失败 + 1 个 guard 断言,属 NIXL 健壮性后续项;该数字是 HiSparse on/off 对比而非本 PR diff 前后对比,硬件为多节点 PD 分离集群(具体 GPU 型号未在正文注明)。

## 验证与回归风险

验证:H200 registered DSV4 NIXL CI 在本 PR 早期修订上通过;GLM-5.1 FP8 TP8/DP8 NIXL+HiSparse smoke 通过;A/B 双臂均 256/256 完成。风险面:`_homogeneous_kv_mem_kind` 对 hetero-TP + 非 VRAM 目标显式 NotImplementedError,TP 异构直传 host 尚未支持;mixed 分段的 kvpart 通知协议依赖 8 字段 tag 解析,通知丢失/乱序需 multipart 状态机兜底;bootstrap 消息新增两个字段依赖长度探测做向后兼容,旧 prefill/新 decode 组合需走默认 VRAM 路径;swap_in_block_size 改默认值可能影响既有 HiSparse 用户性能(1024→512)。

## 关联

同区域(PD 分离传输):本批次 #23958(PD 分离 metadata buffer,同样以 NIXL 为测试后端)与之交叉;pack 未列出其他直接关联 PR 编号(Closes: DYN-3220 为内部工单)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
