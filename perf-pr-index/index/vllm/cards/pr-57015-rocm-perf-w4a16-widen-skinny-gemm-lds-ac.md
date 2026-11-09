# PR #57015: [ROCm][Perf] W4A16: widen skinny GEMM LDS activation loads to b128

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roberteg16 |
| 合并 | 2026-10-27 |
| 分类 | Attention / Kernel |
| Label | rocm, ready |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/57015 |

## 一句话结论

向编译器证明 int4 skinny GEMM 的 LDS activation 读取对齐,把逐行加载拓宽为 b128,小 N 形状 GEMM 提速最高 17%。

## 动机

`wvSplitK` int4 skinny GEMM 把 activation 以 `s[k_ + K * n]` 存入 LDS 再逐行读回。`K` 是运行期值,编译器无法证明每行偏移 `K*n` 满足 16 元素对齐,较高的行只能回退到成对的 `ds_load_2addr_b32`/`b64` 慢速加载。但 `K` 是 `A_CHUNK`(由 launch macro 固定为 16)的倍数、`k_` 也是,所以 `s[k_ + K * n]` 对每一行实际都按 `sizeof(bigTypeA)` 对齐——这一不变量此前是隐式的,k-loop 本身也依赖它(整块消费 `A_CHUNK`)。

## 优化点

本 PR 把隐式对齐不变量显式化给编译器:读回处改用 `__builtin_assume_aligned(&s[k_ + K * n], kActLdsAlign)`,使编译器把所有 activation 行拓宽为 `ds_load_b128`;LDS buffer 声明加 `alignas(sizeof(bigTypeA))`;host 侧 `K % 16 == 0` 检查改用共享常量 `kInt4KAlign = 16`,每个 kernel 加 `static_assert(kInt4KAlign % A_CHUNK == 0)`,保证将来调大 `A_CHUNK` 时若不同步调大 `kInt4KAlign` 会直接编译失败。这是编译器可见性论证而非架构特性,不针对特定 gfx target;medium kernel 中对全局 `A[]` 的 fallback 读取刻意不动(其对齐无保证)。

## 关键改动

```text
csrc/rocm/skinny_gemms_int4.cu: LDS activation 读取加 __builtin_assume_aligned + alignas + static_assert 硬化
```

```python
bigA[n][k2] = *((const bigTypeA*)__builtin_assume_aligned(
    &(s[k_ + K * n]), kActLdsAlign));
```

## 性能声明

ISA 层面(gfx1151 + ROCm 10.2,反汇编设备镜像统计):LDS load 从 6864 条 `ds_load_b128` + 3744 条 `2addr_b32` + 624 条 `2addr_b64`(共 11232)变为 9360 条全 `b128`(共 9360);N=1/2 本就完全拓宽,改动限于 N=3,4,5。kernel benchmark(`benchmark_rdna_hybrid_w4a16_gemm.py`,group-size 128,batch 4,3 次取中位,TFLOP/s 对称/非对称):2048x2048 +17.1%/+14.6%,2560x2560 +11.6%/+6.5%,8192x2048 +2.6%/+6.0%,2048x6144 -0.8%/-1.4%,大 N 形状基本持平;收益集中在小 out_features(LDS 读受限),batch 1/2 无变化。只有 `M<=5 且 K*M<=32768` 的形状走 HIP skinny 路径,其余路由 Triton 不受影响。模型级:GSM8K 5-shot(Qwen3-8B-AWQ,gfx1151)strict 0.8582→0.8590,平均输出吞吐 20.9 vs 20.6 tok/s 持平。

## 验证与回归风险

`tests/kernels/quantization/test_rdna_hybrid_w4a16.py` 84 passed;ISA 经反汇编逐 instantiation 验证且硬化改动 byte-identical;`A_CHUNK=32` 时 build 按 static_assert 预期失败、调大 `kInt4KAlign` 后恢复。风险面限于 ROCm int4 skinny GEMM 路径;非对称列 benchmark 依赖 #57004(缺则 `hybrid-w4a16-zp` provider 报错)。

## 关联

依赖:#57004(零点处理,benchmark asymmetric 列所需)。同文件另一 open PR:#56301(host 侧 weight row-stride padding,不碰 LDS activation 读取)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
