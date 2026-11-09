# PR #60096: [Model][MoE] Load MegaMoE experts directly into the DeepGEMM layout

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-10-13 |
| 分类 | MoE(tier: major) |
| Label | ready, deepseek, quantization, DSv4, dflash, kimi, k3, DSv4.1 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 可用但另有 8 个文件因超长未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/60096 |

## 一句话结论

DeepSeek-V4/Kimi-K3 的 MegaMoE 专家权重在加载时直接写入 DeepGEMM kernel 布局,删除 post-load transform,修复 `reload_weights` 后精度崩塌(0.948→0.008)。

## 动机

#58053 引入的 DeepGemmMegaMoEBackend 把布局转换放在加载之后:checkpoint 先装入 loader 侧参数,`finalize_weights` 再调 DeepGEMM 的 `transform_weights_for_mega_moe` 转换成 kernel 布局并丢弃原参数。后果有二:任何后续 `load_weights`(热重载、权重更新)无处可写——DeepSeek-V4-Flash 在 reload 后 GSM8K 从 0.948 崩到 0.008,Kimi-K3 在 main 上"幸存"只是因为 `finalize_weights` 见已转换专家就提前返回,reload 的权重根本没进 kernel;post-load 换存储与原地权重更新天然互斥。共享专家融合也同理发生在 load 之后,布局可能在加载后突变。

## 优化点

把布局决策前移到构造期:backend 在 experts 构造时选定(meta 设备 / `--load-format ipc_cache` 时取 worker 当前 GPU),`create_weights` 直接按 kernel 布局注册参数并作为唯一存储——SM100 布局下 `w13_weight` 为 (E, 2i, h//2) int8(FP4 每字节两权重),scale 为 UE8M0 每 1x32 块一个、沿 K 每 int32 打包 4 个、MN-major(构造时即以 `transpose` 形成 MN-major stride);`load_weight(dst, src, shard_id, is_scale)` 把每个 checkpoint shard 写进最终位置的 strided view:gate 行 r → `(r//8)*16 + r%8`、up 行 r → `(r//8)*16 + 8 + r%8`(8 行一组交错),scale 走 UTCCP 行映射 `128b+32j+i → 128b+4i+j`。于是 `finalize_weights`/`finalize_mega_moe_weights` 与仅调用它们的 post-load 钩子全部删除(DeepSeek-V4 保留 mHC 钩子),reload 原地覆盖、不换存储。共享专家融合同样在构造期由 `supports_shared_experts(shared_experts)` 决定(要求 FP8 权重 + 可 tile 到 1x32 的 UE8M0 scale + DeepGEMM 带 shared-expert API),融合时 `DeepseekV4MoE.shared_experts=None`、`shared_experts.*` 张量直接路由到 experts 的 4 个 kernel 参数;不满足则警告一次并保留独立 shared MLP。Kimi-K3 相应删掉 `finalize_weights` 覆写与 `_mega_l*` 缓冲。与 #59003(经 reload 生命周期 staging + 重打包)思路相反,本 PR 直接消灭 packed copy。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py: backend 接口改为 create_weights/load_weight/supports_shared_experts
(其余 model/loader 侧文件在未收录的 8 个 diff 文件中)
tests/models/test_deepseek_v4_mega_moe.py: 布局对照/reload 原地性/融合判定测试重写
tests/models/kimi_k3/test_weight_loading.py: 删除(只校验被移除的 finalize)
```

SM100 布局注册(节选):

```python
register("w13_weight", torch.zeros(e, 2*i, h//2, dtype=torch.int8))
register("w13_weight_scale",
         torch.zeros(e, h//128, 2*i, dtype=torch.int32).transpose(1, 2))
register("w2_weight", torch.zeros(e, h, i//2, dtype=torch.int8))
register("w2_weight_scale",
         torch.zeros(e, i//128, h, dtype=torch.int32).transpose(1, 2))
```

## 性能声明

本 PR 是正确性/内存/加载路径修复,无吞吐提升声明。PR Claims 中与性能相关的数据:identity reload 不改变任何参数、不移动存储(DeepSeek-V4-Flash TP2,1588 个 per-rank checksum 全部 0 diff);共享专家无第二份拷贝,显存 76.6 GiB 与 main 持平。Validation(GB200,真实权重,`vllm serve --moe-backend deep_gemm_mega_moe`,EP;GSM8K 1319 题 5-shot greedy):DeepSeek-V4-Flash TP2 MTP k=2:main 加载后 0.948 → reload 后 0.008,PR 为 0.951 → 0.948;DeepSeek-V4.1-Flash TP4:main 0.898 → 0.010,PR 0.904 → 0.901;Kimi-K3(1.5T MXFP4,4 节点 TP4×DP4、EP16):main 0.944 → 0.950(137.52 GiB),PR 两次 reload 分别 0.949 → 0.937 与 0.945 → 0.954(137.52 GiB)。加载/转换耗时未量化。

## 验证与回归风险

`tests/models/test_deepseek_v4_mega_moe.py` 在 GB200 上 68 个测试全过:乱序 per-shard 加载(含 EPLB replica)对照 DeepGEMM 批量 transform(有 DeepGEMM 时含真实 CUDA transform)、二次加载原地更新(data_ptr 不变)、共享专家 shard(.w2./.down_proj. 双命名、gate/up/down 与 1x32 scale)落位与单 shard 原地重载、构造期融合判定 6 组参数化(旧 DeepGEMM/BF16/非 UE8M0/不 tile 1x32 均不融合)。pre-commit 通过;经 Codex/Kimi K3/Claude 审计,两个 should-findings(共享专家以 checkpoint 名上报 weight tracking)已修复并复验。回归风险:`transform_weights`/`transform_shared_expert_weights` 后端接口被替换,基于 #58053 的 out-of-tree 后端需要迁移;EPLB 在 rebalance 后 reload 的场景 main 上即已损坏,本 PR 未处理;XPU 保留独立实现;`process_weights_after_loading` 的 MTP/DSpark drafter mega-moe 钩子移除,相关测试同步改写。

## 关联

前序: #58053(DeepGemmMegaMoEBackend 与 post-load transform 的引入);替代方案: #59003(draft,基于 staging 的 reload 修复)。适用模型:DeepSeek-V4/V4.1、Kimi-K3(标签 DSv4/dflash/k3)。本批次内无其他 MegaMoE 卡。
