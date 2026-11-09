# RFC #47095: [RFC]: Extension points for backends whose physical layout isn't a permutation of the logical layout (weights, quant scales, KV-cache, MLA, EP)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47095](https://github.com/vllm-project/vllm/issues/47095) |
| 作者 | ChuanLi1101 |
| 创建日期 | 2026-06-30 |
| 状态 | ⚪ closed (2026-10-31) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |

## 一句话结论
梳理「物理布局非逻辑布局排列」后端(以 AMD AITER 为例)所需的 6 个扩展点,建议并入 #42082/#8913/#33314 三个在途工作而非另起炉灶。

## 提案动机
vLLM 已有三个在途努力:#42082(标准 KV 布局,`stride_order` 排列)、#8913(量化格式与内核解耦)、#33314(LinearParams + `process_weights_after_loading` repack hook),已覆盖线性权重、量化 scale 及 KV 的可排列部分。但集成 AITER(CK/ASM/Triton 内核)目前只能靠 translation shim:加载时 reshape/shuffle、每次 forward 重新 view/permute,脆弱且耗内存/延迟,还重写了 vLLM 自己的 TP 分片、账目与块管理逻辑。缺口对任何原生布局真正不同的后端都存在。

## 提出的方案
6 个扩展点:
1. KV 布局超出 stride 排列:头维分解微铺(reshape 非排列)、K/V 异构布局、scheduler 与 kernel block size 不一致需虚/实块表翻译(扩 #42082)。
2. MoE 权重/scale repack hook(#33314 明确不含 FusedMoE,而 AITER 的 repack 大头在 MoE)。
3. 量化 scale 新增可声明轴:orientation(行/列主序)、dtype(UE8M0/uint8 微缩放)、FP8 变体 `e4m3fn`/`e4m3fnuz`(扩 #8913/#33314)。
4. 对齐/填充作为声明属性(物理 padded shape ≠ 逻辑 shape,免去 per-model glue)。
5. MLA paging:`page_size=1` 线性索引、辅助 per-layer 缓存(如 sparse FP8 index cache)、last-page-length 元数据(扩 #42082)。
6. EP dispatch contract:静态大小 dispatch buffer、per-local-expert token 计数、int32 top-k、声明不支持的 routing modes。

## 优化点与收益
正文未给出量化数字;收益在架构层面:用框架统一接管布局/分片/块管理,消除 per-backend shim 的内存与延迟开销。作者表示可按区域拆成小 RFC 分别并入。

## 现状与落地
已关闭(2026-10-31),labels 含 stale;正文看不出各 ask 是否被吸收,落地情况需查 #42082/#8913/#33314 的后续进展。

## 关联
#42082、#8913、#33314、ROCm/aiter#3885、PR #42374

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
