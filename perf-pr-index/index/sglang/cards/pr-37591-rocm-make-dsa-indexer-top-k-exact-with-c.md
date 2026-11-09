# PR #37591: [ROCm] Make DSA indexer top-k exact with cooperative selection

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiejingzhangamd |
| 合并 | 2026-09-06 |
| 分类 | Attention / Kernel | tier: major |
| Label | amd, sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 展示 3 个文件(`rocm_hipify.py`、新增 `topk.hip`、新增 `dsa_topk_coop.cuh`),另有 3 个文件因超长从 diff 中略去 |
| 链接 | https://github.com/sgl-project/sglang/pull/37591 |

## 一句话结论

用原生 HIP cooperative top-k 替换 hipify 生成的 ROCm DSA indexer top-k,修复精度丢失(recall 0.53→1.0)并在 MI355X 上最高约 3x 提速。

## 动机

PR 之前 ROCm 路径使用由 `topk.cu` hipify 生成的 `topk.hip`。其 fp16 coarse key 在真实 indexer 分数分布下会把一行的大部分坍缩进同一 bucket:从 GLM-5.2-MXFP4 ROCm decode 捕获的 logits(20 行/次,context 至 134,849 token,分数约 0.04-180)中,8-bit fp16 key 只填出 4-126 个 bin,最大 bin 占一行 6%-88%;且 candidate buffer 溢出时会静默丢弃合法 winner。结果是在捕获的 GLM-5.2 logits 上 selected-value recall 对 `torch.topk` 低至 0.53(六次捕获调用分别为 0.53/0.97/0.97/1.00/1.00/1.00,prefill chunk 0.89)——最差一次丢弃了近一半 indexer 应找到的 KV。top-k 选择错误直接决定 DSA indexer 检索哪些 KV,值得保证精确。

## 优化点

新写原生 HIP kernel(形态借鉴 vLLM `cooperative_topk.cuh`,后者署名来自 sglang#23600,但 thread block clusters/dsmem/mbarrier/TMA 无 ROCm 等价物且 CDNA warp 64 lane,代码全部重写):(1) coarse histogram 用 16-bit ordered fp32 key(sign + 8 exponent bits + 7 mantissa bits),拆成 12-bit round 0 与按需 4-bit round 1,单调且尺度不变,把捕获行的 populated bucket 从 4-126 提升到 6-243;(2) 阈值 bin 以上的元素直接判为 winner,阈值 bin 成员进 LDS tie buffer(4096),随后对该集合做 4 轮 radix-256 的 fp32 精确 refine;(3) tie buffer 放不下时按已解析 prefix 重扫整行重建候选集,使可并列元素数无上限、结果仍精确——这是与旧 kernel(截断候选集)的本质区别;(4) `hist_add_aggregated` 在同 wave 全部 lane 命中同一 bin 时合并为一次 atomic,clustered 分布下 100500 长度 one-block 路径 270.5 us → 135.8 us;(5) row splitting:batch 不足以填满机器时(行长 ≥65536、每行 ≥4 块、≤32 块)把一行拆到多 block,分摊 LDS histogram 争用并增加内存并行,决策只用 host 可见的 row stride 以保证 capture/replay 一致。CUDA 实现保持不变,ROCm build 不再 hipify `topk.cu`,改为直接跟踪 `topk.hip`;三个入口(`fast_topk`、`fast_topk_transform_fused`、`fast_topk_transform_ragged_fused`)的 raw/paged/ragged 接口与语义不变,并强制在 score 所在 device 上 launch。

## 关键改动

```text
python/sglang/kernels/aot/csrc/elementwise/topk.hip: 新增;ROCm 三个入口实现 + row_split 启发式与 launch 门控
python/sglang/kernels/aot/include/hip/dsa_topk_coop.cuh: 新增;cooperative top-k kernel(直方图/散射/radix refine/行拆分)
3rdparty/amd/wheel/sgl-kernel/rocm_hipify.py: 从 hipify 列表移除 topk.cu,ROCm 改用跟踪的原生 topk.hip
```

```cpp
constexpr uint32_t kTopK = 2048;
// 12-bit bins (16 KB) plus 4096 ties (32 KB) is 48 KB of LDS,
// inside the 64 KB static ceiling that gfx942 imposes.
constexpr uint32_t kHistBits = 12;
constexpr uint32_t kTieCap = 4096;
constexpr uint32_t kBlock = 1024;
...
if (g == 0) {
  coop_topk_kernel<kTopK, kHistBits, kTieCap, kBlock><<<batch, kBlock, 0, stream>>>(p);
  return;
}
```

## 性能声明

基准条件:MI355X (gfx950)、graph replay、width 131,072、100,500 live positions、top-k 2,048、page size 1,时间为每次调用微秒。

- 对比原 hipify 生成的 `topk.hip`:tiny 分布 1.94x-3.18x(b1:95.2→31.6 us);diffuse 1.54x-1.79x(b1:54.3→31.6 us);clustered 1.07x-1.60x(b1:160.9→102.2 us)。在捕获的 GLM-5.2 decode shapes 上,row splitting 相对原生成 kernel 额外 2.1-2.8x。
- 对比现有 JIT `topk_v2.cuh`(同条件):tiny 分数下 native 1.20-2.43x 更快;diffuse 在 batch ≤14 时 native 1.05-1.10x 更快,batch ≥32 起 `topk_v2` 反超 1.04-1.49x;更短行(4K-65K live positions)`topk_v2` 1.24-2.96x 更快。但作者强调该对比非等价:`topk_v2` 用 fp16 coarse histogram 且阈值 bin 候选 buffer 上限 2,048 无溢出重扫,selected-value recall 在 tiny/clustered 分布仅 0.0420/0.0220(native 均为 1.0000),其在高 batch clustered 输入上的"优势"不是合法 like-for-like 提升。
- 端到端服务指标:PR 未提供。

## 验证与回归风险

测试:gfx950 ROCm `sgl-kernel` 构建用跟踪的原生 `topk.hip`;`pytest python/sglang/kernels/aot/tests/test_topk.py -q` 161 passed;pre-commit 全部改动文件通过;覆盖 mixed-device launch 与 rejection、one-block 与 row-split 两条路径。风险:仅影响 ROCm(gfx942/gfx950)构建,CUDA 路径不变;top-k 以模板参数固定为 2048(DeepSeek-V3.2 与 GLM-5.2 `index_topk` 的取值),其他 k 值不支持;并列分数按到达顺序选取,跨 run 非确定(与被替换 kernel 及 CUDA 路径一致,测试有 MAX_PERMIT_ERROR);row-split 决策基于 score buffer 的 row stride 而非真实序列长度,极宽 buffer 装短行时会有固定代价(实测 32768/b1 案例 +18%,但该案例本已赢 2x);`SGL_DSA_TOPK_ROW_SPLIT` 环境变量可强制/禁用拆分。

## 关联

sglang#23600(vLLM `cooperative_topk.cuh` 引用的算法形态来源);对照物为 JIT `topk_v2.cuh`(同区域 DeepSeek-V4 top-k 实现,精度不等价)。无正文列出的其他同分类 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
