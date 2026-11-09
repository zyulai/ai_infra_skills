# PR #34580: [AMD] Optimize KIMI-K3 with Triton MLA decode kernel by tuning the stage-1 geometry for gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-danli103 |
| 合并 | 2026-08-18 |
| 分类 | Attention / Kernel(tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat |
| 链接 | https://github.com/sgl-project/sglang/pull/34580 |

## 一句话结论

按 gfx950 重新调优 Triton MLA decode 的 stage-1 几何(BLOCK_N、warps、KV split 数),Kimi-K3 68k 长上下文 ITL 提升 46–73%,默认关闭、env 开关开启。

## 动机

Triton MLA decode kernel 的 stage-1 几何(KV split 数与 KV 轴 block size)取自 CDNA3 上调优的常量,在 gfx950 上不适配:split 数由固定 workgroup 预算推导,小 batch 时单个 split 覆盖整条序列(CU 填不满),大 batch 时预算被当成舍入目标而非上限,stage-1 启动的 workgroup 数超过设备一个 wave 能消化的量,多付整整一个 wave。

## 优化点

全部改动在 `decode_attention.py`,经 `SGLANG_MLA_DECODE_TUNE`(默认 off)opt-in:(1) tuned 路径统一 BLOCK_N=32(HIP 路径原为 16——16 时首个 dot 只有单个 16×16 MFMA tile,warps 只能沿 K=576 切并每步付跨 warp 归约;64 实测不占优);(2) num_warps/num_stages/workgroup 预算按 batch 分桶(batch≤5:4 warps/上限 112 splits;≤24:2 warps/256;其余 1 warp/1 stage),预算=`core_count×4/num_warps` 且用 floor 当上限,永不越过(batch 24、68k 上下文:21 splits 358 us,22 splits 528 us);batch<6 时 split 数按实测上限截断而非按预算缩放。(3) split 数同时下发两个 stage:stage-1 作 runtime 参数(不进 constexpr,避免 CUDA graph 阶梯多编译),stage-2 作 constexpr(是循环展开边界);由 `_mla_launch_plan` 在拥有两次 launch 的调用方一次性定夺,防止两 stage 计数不一致导致静默错读。(4) 确定性:`num_warps` 会重排 fp32 累加,`--enable-deterministic-inference` 或显式 `--triton-attention-split-tile-size` 下保留 scheduler 的每序列 split 数并使用 batch 无关几何(batch invariance)。

## 关键改动

```text
python/sglang/kernels/ops/attention/decode_attention.py: 唯一内核改动文件——分桶几何、split 预算规则、_mla_launch_plan、stage1/stage2 接受 forced_kv_splits
python/sglang/srt/environ.py: 新增 SGLANG_MLA_DECODE_TUNE(EnvBool,默认 False)
test/registered/unit/layers/attention/test_mla_decode_geometry.py: CPU 单测(17 passed, 2079 subtests)
test/registered/unit/layers/attention/test_mla_decode_forced_splits.py: AMD GPU(gfx95)单测(4 passed, 16 subtests)
```

```python
_MLA_BUCKETS = (
    _MlaBucket(num_warps=4, num_stages=2, max_splits=112, batch_max=5),
    _MlaBucket(num_warps=2, num_stages=2, max_splits=256, batch_max=24),
    _MlaBucket(num_warps=1, num_stages=1, max_splits=256),
)

def _mla_kv_splits(batch, head_tiles, max_kv_splits, core_count):
    if core_count <= 0:
        return 0
    bucket = _mla_bucket(batch)
    budget = _mla_split_budget(bucket.num_warps, core_count)
    splits = min(max_kv_splits, bucket.max_splits,
                 budget // max(1, batch * head_tiles))
    return max(1, splits)
```

## 性能声明

硬件 MI355X TP8,模型 Kimi-K3,基线为 0805 docker image(0812 image、triton 3.7.0/torch 2.9.1+rocm7.2 复测)。长上下文 68,086 in/350 out:prefix caching on、8 条共享 prefix、每并发点冷 prefix,aiperf 0.11.0(并发 6/12/24/32),在 FP8 KV cache(fp8_e4m3)+ aiter MHA prefill 之上隔离本内核贡献,ITL 提升 46–73%。短上下文 8192 in/1024 out:bf16 KV、page_size=1、triton 前后端、radix off,MLA 仅占 decode step 的 7.5%(conc 32 时 2.68/35.72 ms),ITL 提升 2–12%,TTT 提升 2–8%。精度:GSM8K(1319)tuning off 0.953 / on 0.955,同配置重复落在 0.951–0.957,精度不变。

## 验证与回归风险

两个新单测:CPU 侧 `test_mla_decode_geometry.py` 钉死 16 个实测 batch 的 split 规则、预算不越界、随 CPX 分区缩放等;GPU 侧 `test_mla_decode_forced_splits.py` 以 2 bf16 ULP(rtol)+ 3e-3(atol)容差验证 tuned 对齐 stock,并断言两 stage 拿同一 count,用三个 mutant 验证测试有效性(TestKeepSchedulerSplits 验证确定性配置走 config 路径)。风险面:默认关闭;开启后 fp32 累加顺序改变(非 bit-equal,正文明确这是 batch invariance 而非与 stock bit 相等);gate 限定 `_is_hip and has_mla and head_dim==576 and is_gfx95_supported() and env`,其余硬件/head_dim/非 MLA 路径不受影响;CUDA graph 阶梯无额外编译(stage-1 计数保持 runtime 参数)。

## 关联

无(素材未引用其他 PR 编号;为 Kimi-K3 在 MI355X/TP8 上的针对性调优,与 aiter prefill、FP8 KV cache 等 AMD 优化栈叠加测得)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
