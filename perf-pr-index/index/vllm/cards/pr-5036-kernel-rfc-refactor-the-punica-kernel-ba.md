# PR #5036: [Kernel][RFC] Refactor the punica kernel based on Triton

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2024-08-01 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未记录);pack 内 diff 仅保留删除的 csrc/punica 源文件,39 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/5036 |

## 一句话结论

把多 LoRA 服务的 punica bgmv CUDA kernel 重写为 Triton SGMV/BGMV kernel,解决性能回退、维度覆盖受限与仅支持 CC≥8.0 的问题。

## 动机

vLLM 的 Multi-LoRA(PR #1804)依赖 punica 的 bgmv CUDA kernel,但存在一组长期问题:性能回退(issue #2829);每支持一个新 head 维度都要改 C++ 宏配置(PR #4007/#4053/#4063 都属此类);部分维度完全无法支持——如 ChatGLM3 在 TP=4 下 ffn_hidden_size 为 3424,不能被 64 整除导致 `static_assert` 失败(issue #3793、#4708);仅支持 compute capability ≥ 8.0 的 GPU;此外编译时间、wheel 体积,以及长上下文下 kernel grid 第二维可能超过 65535 导致 launch 失败,都是痛点。

## 优化点

机制层面是用 Triton 重写 punica kernel:Triton 提供硬件无关的 GPU 编程方式,同时支持 NVIDIA 与 AMD,新增维度不再需要逐个实例化 CUDA 模板。具体实现:shrink kernel 基于 GroupedGEMM 并引入 SPLIT-K;expand kernel 基于 GroupedGEMM;用 SGMV Triton kernel 替换原 punica bgmv kernel,并同时实现了 SGMV(shrink/expand)与 BGMV(shrink/expand)两套。工作计划中的 SGMV 实现、bgmv 替换、单元测试三项均已完成;kernel 性能调优与 auto-tuning 配置留待后续 PR。

## 关键改动

```text
csrc/punica/bgmv/*(bgmv_config.h、bgmv_impl.cuh、各 dtype 的 .cu 等):
  整体删除,连同 punica LICENSE
vllm/lora 下新增 Triton SGMV/BGMV kernel 与集成代码(pack diff 因篇幅省略,
  具体文件名未收入素材)
tests: 新增 sgmv triton kernel 单元测试(文件名未收入素材)
```

注:pack 保留的 diff 只有被删除的 CUDA punica 文件(如 `bgmv_shrink_kernel`/`bgmv_expand_kernel` 的实现),新增 Triton 侧文件未包含在素材中。

## 性能声明

PR 正文未给出具体数字:仅声明 "the above issues were addressed or alleviated(performance drop)",并附两张 benchmark 截图(图片内容未收入 pack 素材,无法引用数值)。

## 验证与回归风险

按作者 06/27 的总结,已完成 Triton kernel 的集成与单元测试,并在 TP(2/4/8)、CUDA Graph、Quantization、Fully sharded LoRA、NVIDIA T4 等场景下验证结果正确。风险点:作者自述这"只是一个开始",kernel 性能调优与 auto-tuning 尚未完成,初期性能可能不及调优后的 CUDA kernel;SGMV/BGMV 之间的适用边界(通常与 batch 大小相关)在正文中未展开;T4 级旧卡可用性扩大,但其余非 NVIDIA 平台仅是"具备可能"。

## 关联

- 前序:PR #1804(Multi-LoRA 支持,引入 punica kernel)
- 修复的问题:issue #2829(性能回退)、#3793 与 #4708(3424 等维度 static_assert 失败)
- 关联配置类 PR:#4007、#4053、#4063(为本 PR 消除的"每加维度改宏"工作流)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
