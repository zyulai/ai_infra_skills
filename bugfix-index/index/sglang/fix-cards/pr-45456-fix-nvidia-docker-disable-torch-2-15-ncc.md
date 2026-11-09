# PR #45456: [Fix] [NVIDIA] docker: Disable torch 2.15 nccl2 in CUDA 13.4 container to fix EPLB rebalance crash

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#45456](https://github.com/sgl-project/sglang/pull/45456) |
| 作者 | trevor-m |
| 合并日期 | 2026-10-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Parallelism / Communication |
| bug 类型 | crash |

## 一句话结论
CUDA 13.4 容器升级 torch 2.15 nightly 后默认启用新 nccl2 后端,EPLB rebalance 报 "ValueError: Tensors must be contiguous" 崩溃;容器内设 `TORCH_DIST_USE_NCCL2=0` 回退旧 nccl 后端。

## 问题现象
在 CUDA 13.4 容器内执行 EPLB rebalance 时崩溃,报错为:

```
ValueError: Tensors must be contiguous
```

触发条件:CUDA 13.4 镜像配上 #45252 升级到的新版 torch 2.15 nightly——torch 已把默认分布式后端切换为新的 "nccl2"。影响面为使用该镜像跑 EPLB(专家负载均衡)rebalance 的部署;除此之外正文未描述其他受影响场景。

## 根因
正文披露:#45252 将 CUDA 13.4 镜像的 torch 2.15 nightly 升级到新版本,torch 默认改用新的 "nccl2" 后端,该后端在 EPLB rebalance 过程中触发 "Tensors must be contiguous" 的 ValueError。正文未进一步披露 nccl2 后端内部为何要求连续张量(如具体哪个集合通信算子、哪个张量不连续),只确认问题由默认后端切换引入。

## 修复方案
在 `docker/Dockerfile.cu134` 的两处镜像 stage 中新增环境变量,关闭 torch.distributed 的 NCCL2 路径、回到升级前使用的 torch nccl 后端:

```diff
+# Turn off torch.distributed's NCCL2 path in the torch nightly.
+ENV TORCH_DIST_USE_NCCL2=0
```

不改动任何 Python/CUDA 代码,仅容器环境变量层面的回退。

## 影响与回归风险
- 影响窗口:CUDA 13.4 容器 + #45252 引入的 torch 2.15 nightly 版本(正文以 #45252 为界,未给出确切的镜像 tag 范围)。
- 回归面:仅 `docker/Dockerfile.cu134`。该容器内所有 torch.distributed 集合通信都会走回旧 nccl 后端——不仅限于 EPLB rebalance;若旧 nccl 后端在该 nightly 上存在其他兼容性问题,会在其他通信路径暴露(正文未评估)。非 CUDA 13.4 镜像与其他容器不受影响。torch 上游把 nccl2 修好或默认稳定后,此环境变量需要重新评估(正文未提及后续计划)。

## 关联
- sgl-project/sglang#45252(引入 torch 2.15 nightly 升级的 PR,正文提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
