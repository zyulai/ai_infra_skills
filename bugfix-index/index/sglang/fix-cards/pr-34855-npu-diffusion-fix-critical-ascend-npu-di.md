# PR #34855: [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34855](https://github.com/sgl-project/sglang/pull/34855) |
| 作者 | OrangeRedeng |
| 合并日期 | 2026-08-22 |
| labels | diffusion, jit-kernel, npu, run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
修复 NPU diffusion 的 Ring attention 缺失、SRT CUDA 兼容层泄漏、MOVA 非连续张量与内存统计崩溃,恢复 2-NPU CI。

## 问题现象
#35004 之后 diffusion 可导入 SRT CLIP 代码并执行 SRT `init_npu_backend()`:在原生 NPU diffusion 进程里导入 `torch_npu.contrib.transfer_to_npu` 会全局改写面向 CUDA 的 API 与设备参数,改变运行时特性检测与 kernel 选择;`allow_internal_format=True` 已知拖慢未量化 diffusion——#20533 的 Wan 回归把 DecodingStage 从 6.1889 s 增至 11.76 s(+90%,慢 1.90 倍)。另有 MOVA 多 NPU 下 attention 返回非连续张量出错、#35641 后 host 内存统计可因 storage 异常失败、2-NPU CI 套件此前被禁用等多处回归。

## 根因
正文披露:
- 运行时回归根因是 SRT 的 CUDA 兼容层(transfer_to_npu 导入与 allow_internal_format=True)泄漏进 diffusion 进程;
- MOVA 正确性根因是 #31453 留下的连续布局假设,多 NPU attention 可能返回非连续张量,view() 失败;
- 内存统计根因是 #35641 引入的基于 storage 的模块大小统计,storage 访问抛 RuntimeError 时会中断内存规划;
- packed varlen/Ring attention 在 NPU 上缺少实现(#33327 扩展 Qwen 到 tail-padded Ring 后暴露该共享路径)。

## 修复方案
- ascend_fa.py 新增基于 `npu_fused_infer_attention_score`(TND)的 packed varlen 实现(独立 Q/K packed 长度、边界校验、空/非连续处理、LSE 归一化)与 `AscendFlashAttnImpl.forward_varlen`;attention 后端新增 `forward_ring_kv_chunk` 接口,usp.py 的 `_ring_attention_varlen` 改为经 attn_impl 调用而非直接调 flash_attn_varlen_func;
- init_npu_backend:当 `sglang.multimodal_gen` 已加载时不导入 transfer_to_npu、不设 `allow_internal_format=True`(SRT 路径保留原有重 mock);
- mova_video_dit.py:attention 后的 `out.view(b, s, -1)` 改为 `reshape`;
- host_memory_budget.py:storage 访问抛 RuntimeError 的张量跳过而非失败;
- 恢复 2-NPU diffusion CI(linux-aarch64-a3-800t-16),1-NPU/2-NPU 套件各分两分区,GT 任务 checkout 指定 ref、下载失败重试,GT revision 更新为 8e3d717e…;sgl-kernel-npu pin 到 2026.8.13 并安装 attentions wheel;刷新 FLUX.2/ERNIE/LTX-2 NPU 基线(仅 CI 数据,非算法提速)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:multimodal_gen 的 Ring/USP attention 路径(`_ring_attention_varlen` 签名 softmax_scale→attn_impl 波及 minimax_h3 等调用点)、Ascend NPU SRT 后端初始化(仅 diffusion 进程跳过 transfer_to_npu)、MOVA 前向、host 内存预算、NPU CI/GT workflow 与 docker 依赖 pin。

## 关联
- sgl-project/sglang#35004、#20533、#20684、#33569、#33327

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
