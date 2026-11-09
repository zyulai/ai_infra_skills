# PR #46724: [Perf] [NVIDIA] muse-spark-open: Fuse MoE finalize with the 2shot_push all-reduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wookjeHan |
| 合并 | 2026-10-31 |
| 分类 | Speculative Decoding (tier: major) |
| Label | jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,3 个文件,另有 1 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/46724 |

## 一句话结论

Muse Spark MoE finalize+all-reduce 融合 kernel 新增 2shot_push 路径,路径切换处 kernel 提速 1.04-1.40x,decode 每步 E2E 约 0.6-1.8%。

## 动机

Muse Spark 把 MoE finalize 与 custom all-reduce 融合成一个 kernel,但原融合 kernel 只实现了 `1shot_push`,且只在独立 all-reduce 本来就会选 1shot_push 的规模下启用;超过该规模时 MoE 输出要回落到"单独 finalize kernel + 单独 all-reduce launch"两条路径,多一次 kernel launch 与整层输出的完整读写。

## 优化点

在 CuteDSL kernel 中按 `custom_all_reduce.cuh` 的 `2shot_push` 协议实现融合两步交换:phase 1 每个 token 的 bf16 partial push 给自己列切片的 owner rank,owner 做 fp32 按序求和、单次 bf16 舍入后写回输出并把 reduced slice 推给其他 rank 的 phase-2 槽位,phase 2 gather 其他 owner 的切片。输出与融合 1-shot bitwise 一致;复用 communicator 的 push slots 与 epoch controller,无新分配。配套改动:`uses_two_shot_push(M)` 与独立 all-reduce 的选路完全一致(含 override_algo),`two_shot_max_tokens = slot_bytes * world / (4H)`(默认 8MB slot 时 TP4 为 2048 token,且要求 `H % (8*world) == 0`);M 超过 controller 数时一个 block finalize 两个 token(每行线程一套 named barrier, multicast 1-shot 除外);单行融合 block 每次 barrier 对处理 4 个 expert(group_width=4);1-shot/2-shot 共享 copy atom、sentinel 重映射、poll、rank 求和等 helper。

## 关键改动

```text
python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py: 核心,新增 push_token_two_shot/reduce_slice_two_shot/gather_slices_two_shot 与两 token per block、group_width=4 调度。
python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py: 透传 ar_two_shot=workspace.uses_two_shot_push(num_tokens)。
python/sglang/srt/layers/communication/muse_spark_finalize_workspace.py: two_shot_max_tokens 容量计算与 uses_two_shot_push 选路。
```

两 shot 交换复用借来的 push slot(双缓冲按 epoch 奇偶):

```python
# Slot (dst, src) 两半:前半存 src 的 partial,后半存 src 的 reduced slice
self.two_shot_max_tokens = (
    self.slot_bytes * self.world_size // (4 * hidden_dim)
    if hidden_dim % (8 * self.world_size) == 0 else 0
)
```

## 性能声明

Kernel 基准(H=4096、top-8、M=8xc 的 DFlash verify,每层 finalize+all-reduce,us/call,CUDA graph 多次 replay 取中位、最慢 rank):路径发生切换处 SM107 TP4 1.04-1.40x、SM107 TP8 1.12-1.37x(TP8 为 2 节点 MNNVL multicast)、GB300 TP4 1.18-1.39x、GB300 TP8 1.15-1.35x;如 SM107 TP4 c=24 从 11.7us(separate)降到 8.4us(fused 2shot)。未切换处不变;GB300 TP4 c=10 因 four-experts-per-barrier 改动使 fused 1-shot 快 5%。E2E 投影(DFlash 8 draft tokens,SM107 模拟接受长度 6 / GB300 4.05,62 个 MoE 层/步):SM107 每步约 -0.6%~-1.7%,GB300 约 -0.8%~-1.8%。硬件为 SM107(TP4 单机/TP8 双节点)与 GB300(TP4 单托盘/TP8 双托盘)。

## 验证与回归风险

正确性验证较充分:多 GPU harness 下 fused 2-shot vs fused 1-shot bitwise(M=8..1024,两种 router-weight dtype、两种 weight_first);与独立 CustomAllReduceV2 交错执行,SM107 world=2/4 与 8(2 节点 MNNVL)各 80/80 通过,GB300 38/38;压力测试(4000 次 pipelined 调用 + CUDA graph 1000 次 replay 含慢 rank)全部 bitwise PASS。E2E 精度:GSM8K 97.35%-97.65%,AIME26 pass@1 95.62%-97.08%,pass@16 100%,接受长度与 main 持平。单测 checklist 未勾选;pack 中 CI Base/Extra/AMD ROCm 三项均标红。风险:2-shot 需 `H % (8*world)==0` 否则容量为 0(自动回落);finalize push 要求每 lane 8 个 bf16 值、整 warp;与独立 all-reduce 共享 slot/epoch controller,交错与清槽逻辑依赖协议正确性。

## 关联

独立于 PR #46197(其提供 SM107 的 2shot_push 选路表),本 PR 在其之上或 GB300 main 之上生效;同区域 MoE finalize 融合:PR #27720(本批次)。Muse Spark 配合 DFlash 投机解码(PR #22080)使用。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
