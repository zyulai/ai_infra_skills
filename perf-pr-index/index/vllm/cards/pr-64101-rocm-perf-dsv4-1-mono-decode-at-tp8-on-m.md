# PR #64101: [ROCm][Perf][DSv4.1] Mono decode at TP8 on MI355X

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding |
| Label | rocm, ready, ci/build, deepseek, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 mono kernel(plan.py/back.py/gemv.py)、mono_decode.py、测试与 CI 配置 |
| 链接 | https://github.com/vllm-project/vllm/pull/64101 |

## 一句话结论

让 DeepSeek-V4.1-Flash 的 `--monokernel` 单体 decode 支持 TP8(gfx950/MI355X),conc 1 step 时间 -37.5%(8.59 → 5.36 ms)、吞吐 +48%。

## 动机

DSv4.1 mono decode 此前只允许 TP2/TP4,`MonoDecodeLayer._create` 直接拒绝其他 TP;而单机 8 卡用户只能起 TP8。作者论证:conc 1 下 TP8 仍可用 mono——每 rank 只流过 1/8 权重,持久化 K1/K2 流水链长度不变,值得把 FlyDSL 内核在 gfx950 上扩展到 TP8(每 rank 8 个 attention head、routed/shared expert 中间维 288,routed 由 loader pad 到 384)。

## 优化点

关键 bug 在 attention 维度规划:TP8 每 rank 仅 8 个 head,而 `head_tiles` 原为 `heads // HEAD_TILE = 8 // 16 = 0`,split-attention 循环不会发射任何 MFMA(静默产生错误输出)。改动:(1) `head_tiles` 改为 `cdiv(heads, 16)`,新增 `heads_pad` 给出整 tile 数;M=7、ratio 2 下 gfx950 CPU 编译显示 bf16 MFMA 从 0 条变为 32 条,与 TP4 相同;(2) `back.py` 的 split 阶段 LDS 与 stat 偏移按 `heads_pad` 尺寸,padding 列参与计算但从不存储(利用列独立性:每个 head 有自己的 score max 与 PV 列,store/combine 只读 `h < heads`);(3) `gemv.py` row-major FP8 步进加载接受 `K % 64 == 32`(TP8 shared expert down projection,K=288),末步上 lane clamp 进行内,两侧 MFMA 零化;(4) `mono_decode.py` 门控放行 TP8(gfx942 仍仅 TP4);(5) 补 TP8 数值/层测试与 MI355X 8 卡 CI step。

## 关键改动

```text
vllm/models/deepseek_v41/amd/mono/attention/plan.py: head_tiles 上取整、新增 heads_pad
vllm/models/deepseek_v41/amd/mono/attention/back.py: LDS/stat 按 heads_pad 尺寸、padding 列不落盘
vllm/models/deepseek_v41/amd/mono/stages/gemv.py: 接受 K % 64 == 32 的尾块
vllm/models/deepseek_v41/amd/mono_decode.py: TP8 进入 monokernel 门控
tests/models/test_dsv41_mono_{numerics,layers}.py + .buildkite: 新增 TP8 用例与 CI
```

```python
     def __post_init__(self):
-        assert self.tp in (2, 4), self.tp
+        assert self.tp in (2, 4, 8), self.tp
...
     def head_tiles(self) -> int:
-        return self.heads // HEAD_TILE
+        return cdiv(self.heads, HEAD_TILE)
+
+    @property
+    def heads_pad(self) -> int:
+        return self.head_tiles * HEAD_TILE
```

## 性能声明

8x MI355X,DeepSeek-V4.1-Flash,MLPerf conc-1 recipe(DSpark k=6、adaptive verification off、MXFP4 sparse-logits indexer),nightly 5d35ce3934 + 本 PR:

- Decode step(random 32k in / 1k out、ignore-eos、合成 AL 3.87):conc1 8.59 → **5.36 ms(-37.5%)**,392 → 580 tok/s(+48%);conc4 step 10.10 → 8.10 ms(-19.8%),TPOT 3.38 → 2.57 ms(-24%),1014 → 1163 tok/s(+15%)。
- 质量对照(mono on/off):GSM8K-1319 8-shot 0.910 vs 0.923(差异未解释,McNemar p=0.041/0.060);RULER NIAH 32k 0.997 vs 0.998、128k 0.993 vs 0.990。
- 边界:conc 4 下单台 TP8 比两台 TP4 副本吞吐低 26%(1262 vs ~1706 tok/s)——TP8 mono 是 conc-1 配置。CPU 编译:TP8 全部 0 VGPR spill,TP2/TP4 二进制与 main 逐字节一致。

## 验证与回归风险

测试:TP8 数值/层用例(`tp in [2,4,8]`、`(8,8)`/`(8,28)` on cdna4)全部通过,新增可选 `mi355_8` CI step;TP2/TP4 内核二进制与 main 逐字节相同,即非 TP8 路径零风险。风险面:仅 gfx950;mono 内核跨 rank spin-wait 无界(TP4 亦然),8 rank 下单 rank hang 拖垮全 server,bounded wait + fail-stop 留作后续 PR;GSM8K 小幅差异未定性。

## 关联

Follow-up:#63885/#63888/#63890/#63891(launch 数削减)、#63893(tuned K1)、#63892(X8R TP4→TP8),组合后 conc1 达 4.53 ms。同批相关:#56254(DS V4.1 indexer DeepGEMM sparse MQA)、#51967(DSV4 top-k 内核)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
