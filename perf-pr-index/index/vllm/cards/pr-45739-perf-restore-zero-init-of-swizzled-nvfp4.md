# PR #45739: [Perf] Restore zero-init of swizzled NVFP4 scale buffer to recover Blackwell decode throughput

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qiching |
| 合并 | 2026-06-30 |
| 分类 | Quantization（tier: major） |
| Label | ready, verified |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；实际 diff 仅 `vllm/_custom_ops.py` 一处 `empty→zeros` |
| 链接 | https://github.com/vllm-project/vllm/pull/45739 |

## 一句话结论

修复 #42988 把 swizzled NVFP4 scale buffer 从 zeros 改 empty 引入的未初始化读，恢复 Blackwell NVFP4 decode 吞吐（回归时坍塌 87–93%）。

## 动机

#42988（"[Perf] zeros -> empty to remove additional fill"）将 `create_fp4_scale_tensor` 的分配从 `torch.zeros` 改为 `torch.empty`。其中 swizzled scale-factor buffer 形状为 `(round_up(m, 128), round_up(n // 16, 4) // 4)`，在 M（到 128）与 packed-K（到 4）两维都有 padding；NVFP4 量化 kernel（`cvt_fp16_to_fp4`）按 swizzled tiled layout 写 scale，并不写满下游 NVFP4 GEMM 会读到的所有 padded 元素，于是这些 scale factor 未初始化，反量化被破坏，Blackwell（B300/GB300）NVFP4 decode 出现严重回归。

## 优化点

#42988 的部分 revert：只把 swizzled `int32` scale 分配恢复为 `torch.zeros`；#42988 的其余改动有意保留，因为那些 buffer 确实可安全留未初始化——同函数内 non-swizzled `uint8` 分支分配的是精确尺寸 `(m, n // 16)`（无 padding），其 kernel（`cvt_fp16_to_fp4_sf_major`，row-major `sf_out_rowmajor_u8`）完整覆盖它；packed-FP8 scale、TurboQuant `attn_out`、GDN `z_out` 均在使用前被完整覆写。修复依据是一条窄而可直接证明的论断：量化 kernel 留下消费者可见的 padded 元素未写（零初始化能完全恢复正确行为，仅当这些元素确实被读到未初始化值时才可能）。

## 关键改动

```text
vllm/_custom_ops.py: create_fp4_scale_tensor 的 swizzled int32 分配 empty→zeros
```

```python
-        return torch.empty(
+        # Must be zero-initialized: the swizzled scale buffer is padded to
+        # (round_up(m, 128), round_up(scale_n, 4) // 4) but the NVFP4 quant
+        # kernel does not write every padded element that the downstream
+        # NVFP4 GEMM reads.
+        return torch.zeros(
             (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
         )
```

## 性能声明

症状（`con1 isl1000 osl1000`，TP4，`--quantization modelopt_fp4`）：output-token throughput 下跌约 87–93%，输出序列长度从请求的 1000 塌缩到约 65–127，ITL 升高约 8–15 倍。

Per-hunk 隔离（B300，Llama-3.3-70B，con=1）：baseline（全 empty）19.56 tok/s / ITL 50.62 ms / OSL 127.0；只 revert swizzled int32 一行即 135.40 tok/s / 7.35 ms / 873.2；只 revert uint8 或 fp8-util 行无改善（19.49/19.56）；全部 revert 162.29 tok/s / 6.11 ms / 1041.4——证明该行必要且充分。

修复前后（GB300NVL，con=1）：Llama-3.3-70B（dense）18.39→143.25 tok/s，ITL 53.99→6.88 ms，OSL 127.0→997.8；Kimi-K2.5（1T MoE）10.16→148.28 tok/s，ITL 97.61→6.58 ms，OSL 65.8→999.6。

修复后并发扫描（GB300NVL）：Llama C=16/64/128 为 2,020.14/7,305.73/11,478.84 tok/s，Kimi 为 1,047.85/2,770.39/4,913.39 tok/s，两模型 OSL 各档均保持约 1000，无退化。OSL 数值系作者由吞吐近似推导（可略超 1000 上限），有效信号是回归（~65–127）对比健康（~870–1040）。

## 验证与回归风险

- 验证：单行切换（empty↔zeros）+ `vllm serve` + aiperf（TP4，ISL/OSL=1000/1000），per-hunk 隔离排除其他分配分支的影响；并发扫描确认无规模下退化。关联 issue：#45741。
- 回归风险：改动仅一条分配语句；理论风险是 zeros 的额外初始化开销，但该 buffer 是 scale factor 大小的 int32 小分配，PR 实测吞吐即含此开销；其余 empty 分支保留，不影响完全覆写的 buffer。

## 关联

- 引入回归的 PR：#42988（zeros→empty 优化，本 PR 为其部分 revert）；关联 issue #45741。
- 同分类：本批 PR #51473、#28892、#32520、#28603。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
