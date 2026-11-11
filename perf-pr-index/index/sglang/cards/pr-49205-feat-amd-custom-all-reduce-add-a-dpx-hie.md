# PR #49205: [Feat] [AMD] custom-all-reduce: Add a DPX hierarchical 2shot_push to carv2 on MI450 (gfx1250)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw(co-author @wenkaidu) |
| 合并 | 2026-11-09 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(diff 摘录完整可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/49205 |

## 一句话结论

gfx1250 DPX 分层 all-reduce:先 SoC 内配对规约再跨机 2shot,TP16 跨机 all-reduce 比扁平 2shot_push 快 17-31%。

## 动机

#49003 让 carv2 以 TP16 跑在两台 MI455X 上,但把 16 个 rank 当作扁平 peer,`2shot_push` 几乎把整张量在 fabric 上送两次。DPX 下同一 SoC 的两个分区共享 HBM,分区间流量远便宜于 SoC 间:单机 TP2 实测 8 MB 2shot_push 同 SoC 19.5 us vs 跨 SoC 28.7 us。若先把每个 pair 内规约到一半,只有一半张量跨 fabric,每个 SoC 的 fabric 流量约减半。

## 优化点

`hier_2shot_push` 三步:pair reduce-scatter(每 SoC 的分区 p 得到半张量的 pair sum)→ 对 8 个分区 p 做 2shot_push(仅半张量跨 fabric)→ pair all-gather。kernel 侧新增 `pair_push_kernel`(ROCm only):即 2-rank 2shot_push 的一半(归约散射或 all-gather),复用 2shot_push 的 Lamport markers 与 per-block epochs;每次 launch 既推给 peer 又轮询并清掉 peer 发来的数据,相邻两个 stage 的 epoch 交替同步,故可共用 one-chunk slot;kernel 与 dispatch 都在 `#ifdef USE_ROCM` 下,CUDA 构建不变。拓扑:`probe_dpx_pairs` 按各 rank 从 sysfs 读到的 UALink `accel_id` 配对,集体执行;任一 rank 读失败或布局不是精确 pair(SPX、CPX、半 SoC)则全员 None、algo 关闭。workspace:pair 与 cross push plane 是现有 symmetric allocation 末尾的视图,各带 epoch counter,无额外 process group 或 rendezvous;默认 max_size 16 MB 时 hier 覆盖到 32 MB、2shot_push 保留 32-64 MB;max_size 提到 32 MB 时 hier 覆盖全部 2shot_push 区间,扁平 push slot 从 8 MB 缩到 256 KB(TP16 每 rank symmetric memory:默认档 352 MB,32 MB 档 264 MB,#49003 为 256 MB)。TP16 表:`hier_2shot_push` 取 256 KB-64 MB,`1shot_push` 保持到 256 KB;非 32 B 倍数的尺寸退到 2shot_push 或 RCCL。其他 world size、SPX、CUDA、其他 ROCm GPU 行为不变。

## 关键改动

```text
python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh: 新增 pair_push_kernel<T, kAllGather>(ROCm only)与 pair_reduce_scatter/pair_all_gather dispatch
python/sglang/srt/distributed/device_communicators/gfx1250_fabric.py: dpx_pairs / probe_dpx_pairs 按 accel_id 配对
python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py: hier workspace(pair/cross planes)、_hier_all_reduce 三步执行
python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py: TP16 行启用 hier_2shot_push Range(256 KB, 64 MB)
```

```python
def _hier_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
    half = custom_pair_stage(self._pair_obj, input.view(-1), all_gather=False)
    half = custom_all_reduce(
        self._cross_obj, half, algo=AllReduceAlgo.TWO_SHOT_PUSH
    )
    out = custom_pair_stage(self._pair_obj, half, all_gather=True)
    return out.view(input.shape)
```

## 性能声明

TP16 双 DPX 主机,bf16,us/all-reduce(max over ranks,median of 3,CUDA graph,RCCL 2.32.3 fabric DDA):384 KB 28.9→20.3(29.8%)、1 MB 34.0→23.5(30.9%)、8 MB 91.0→74.0(18.7%)、64 MB 429.6→347.6(19.1%),384 KB-64 MB 区间比扁平 2shot_push 快 17-31%;末 rank 迟到 ~20 us 时快 7-15%(64 MB 435.7→372.7,14.5%)。范围选择:128 KB 处 1shot_push 仍更快(13.2 vs 19.4),256 KB 处 hier 反超(20.0 vs 21.4;迟到 37.1 vs 42.0)。叠加后续 8-rank 2shot_push kernel 提速后,8-64 MB 快 21-30%(PR 注明的 follow-up 测量)。

## 验证与回归风险

两台 MI455X(DPX,8 GPU/台,HIP 7.16):CPU 测试 49 通过;单机 test_custom_all_reduce.py 310 通过/world size;TP16 四种算法 × bf16/fp16/fp32 × 16 B-64 MB(含每个阈值边界)eager+100 graph replay 0 失败;max_size 16/32 MB 两档同样 0 失败;TP8(4+4)hier 保持关闭、结果同 #49003;双机 kernel log 0 fabric 事件。新 CPU 测试覆盖非相邻 rank 的配对、单 rank 失败的 probe 一致性、hier sizing 与每个阈值的 dispatch。风险:Muse Spark fused MoE finalize+all-reduce 只在独立 all-reduce 选 2shot_push 时融合,TP16 DPX 的 hier 区间内改为不融合(结果正确,作者未测该模型 TP16);非 32 B 倍数尺寸在 hier 区间上限附近可能落到 RCCL;TP16 symmetric memory 占用增大。

## 关联

#49003(直接前置:跨 UALink 节点 carv2 与 TP16 行);follow-up:8-rank 及以下 2shot_push kernel 提速(@wenkaidu 的独立 PR),hier 的中段自动受益。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
