# PR #32118: Fix nightly CI: NVFP4 cuda-graph crash, NVILA batching, CuTe paged-KV zero-size, Kimi-VL OOM

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32118](https://github.com/sgl-project/sglang/pull/32118) |
| 作者 | b8zhong |
| 合并日期 | 2026-07-29 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论

修 nightly CI 四处崩溃:CuTe paged-KV 零尺寸、NVILA stack 异构报错、Kimi-VL 视觉 OOM(换 VisionAttention)、NVFP4 cuda-graph 崩溃。

## 问题现象

triage Nightly Test (Nvidia) 发现的多个独立失败,每个都有完整 traceback:

1. `nvila.py`:`test_vlms_mmmu_eval.py` 报 `stack expects each tensor to be equal size`。
2. `flash_attn/cute/interface.py`:`test_disaggregation_dwdp_mimo.py` / `test_mimo_cp.py` 在 CuTe FA4 paged-KV kernel 报 `ValueError: Expected size in shape to be strictly positive, but got 0`。
3. `kimi_vl.py` / `kimi_vl_moonvit.py`:`test_vlms_mmmu_eval.py` 在 Kimi-VL 上 OOM。
4. NVFP4 cuda-graph 崩溃(见正文第 1 项,pack 中该项内容缺失,仅在测试计划中提及)。

## 根因

正文逐项披露:

1. NVILA:`get_image_feature` 对各图 patch 数随分辨率不同的 vision feature 张量先 `torch.stack` 再立即 rearrange 压掉堆叠维,本意就是拼接。
2. CuTe paged-KV:SM100 diff-headdim(`head_dim != head_dim_v`,如 192/128)省 SMEM 优化在 split-KV 下把 `n_block_size` 缩到 64;但 `q_stage==1`(decode)且 paged KV 时非 TMA 加载路径用 128 线程,`page_entry_per_thread = n_block_size // num_threads` 向下取整为 0。
3. Kimi-VL:MoonViT 视觉塔强制走 HF 的 `sdpa` attention(对所有打包图像 token 做稠密 attention)而非 sglang 自带 flash-attn,导致 OOM。
4. NVFP4:正文第 1 项在 pack 中缺失;测试计划提到该修复针对 pinned memory,验证方式是直接在真实 `torch.cuda.graph()` 中捕获 `nvfp4_quantize(..., backend="cute-dsl", per_token_activation=True)`,在未修补的 flashinfer 上崩溃、上游修复后成功。

## 修复方案

1. `nvila.py`:`torch.stack` 改为 `torch.cat`,删除多余的 rearrange。
2. `flash_attn/cute/interface.py` 的 `_flash_attn_fwd`:SMEM 缩减条件追加 `not (page_table is not None and q_stage == 1)`,跳过该组合。
3. `kimi_vl_moonvit.py`:MoonViT attention 改经 sglang `VisionAttention` 包装(与 `kimi_k25.py` 同模式),删除自带的 `multihead_attention`/`sdpa_attention`;`kimi_vl.py` 的 `load_weights` 把 checkpoint 的 `wqkv.`/`wo.` 权重名重映射为 `attn.qkv_proj.`/`attn.proj.`,并移除 vision tower 的 `use_tensor_parallel` 透传。
4. NVFP4:依赖 flashinfer 上游修复(pack diff 中无对应 hunk)。

CuTe 修复在真实 B300 上以 MiMo-V2.5 实际配置(`head_dim=192`、`head_dim_v=128`、`page_size=1`、强制 split-KV)独立复现:修复前崩溃,修复后输出有限值且形状正确。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:CuTe 条件收紧只影响 diff-headdim + split-KV + paged KV + decode 的组合(其余仍缩减);NVILA 仅改拼接算子;Kimi-VL 视觉塔结构重写、权重名重映射(单测已验证参数名匹配),DP/TP 相关参数调整;NVFP4 路径依赖外部依赖版本。CI 状态栏 Base/Extra 显示失败状态。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
