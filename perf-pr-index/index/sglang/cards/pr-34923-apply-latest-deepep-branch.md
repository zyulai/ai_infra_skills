# PR #34923: Apply latest DeepEP branch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2026-08-18 |
| 分类 | Speculative Decoding(classify 结果;tier: major) |
| Label | dependencies, deepseek, run-ci, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全量保留(6 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/34923 |

## 一句话结论

适配 DeepEP v2:在 CUDA 低延迟 buffer 初始化前把 NVSHMEM_QP_DEPTH 抬到 dispatch 容量的两倍以上,并强制 NCCL 2.30.7、bump sgl-deep-ep 0.1.1。

## 动机

DeepEP v2 依赖两件事:NCCL 2.30.7 提供的 NCCL Gin;以及 NVSHMEM QP depth 至少为低延迟 dispatch 容量的两倍加二。DeepSeek V4 配置下每 rank 1024 个 dispatch token,所需 QP depth 为 2050,高于 NVSHMEM 默认值 1024——不配置则 CUDA DeepEP 低延迟路径无法正常工作。运行时依赖随之从 `sgl-deep-ep==0.1.0` 升到 `0.1.1`。

## 优化点

(1) `deepep.py` 新增 `_set_nvshmem_qp_depth(num_max_dispatch_tokens_per_rank)`:在 `get_deepep_buffer` 启用 low-latency 模式时(非 NPU)设置 `NVSHMEM_QP_DEPTH = max(现有值, 1024, 2 * (num_max_dispatch_tokens_per_rank + 1))`,保留用户更大的显式取值,NPU 与 normal DeepEP 路径不动;在 buffer 分配前写入环境变量,保证 NVSHMEM 初始化时生效。(2) CUDA 13 环境强制重装 `nvidia-nccl-cu13==2.30.7 --force-reinstall --no-deps`:主 CUDA 13 镜像(Dockerfile 加 `NCCL_VERSION` ARG)与 CI 脚本(`install_nccl` 步骤)双处落地,CUDA 12 显式跳过。(3) 测试侧:DeepSeek-V3 cutedsl 4gpu 测试中 `TestDummyWithSBO` 因当前 DeepEP 不接受 `--enable-single-batch-overlap` 而 skip(待 DeepEP#10 修复)。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/deepep.py: _set_nvshmem_qp_depth(取 max(现有,1024,2N+2))
python/pyproject.toml: sgl-deep-ep==0.1.0 → 0.1.1
docker/Dockerfile: CUDA 13 强装 nvidia-nccl-cu13==2.30.7
scripts/ci/cuda/ci_install_dependency.sh: install_nccl 步骤
```

## 性能声明

无吞吐/延迟对比;验证数据(GB300,4 GPU,`NVSHMEM_QP_DEPTH` 显式 unset):`TestDSV4FlashFP4B200Balanced` 9 个测试全过(165.585 s),GSM8K 0.965,平均投机接受长度 1.9488。PR 未注明除硬件外更细的测试条件(模型为该测试套件的 DSV4-Flash FP4)。

## 验证与回归风险

pre-commit 通过;PyPI 确认 sgl-deep-ep 0.1.1 为最新(8 个 wheel);mock 验证 CUDA 13 安装 NCCL Gin wheel、CUDA 12 跳过。风险面:NVSHMEM_QP_DEPTH 只增不减,若用户显式设置了更大值则保留;NCCL 强装在镜像与 CI 两侧需保持一致,否则 DeepEP v2 的 Gin 依赖缺位;`--enable-single-batch-overlap` 组合被测试 skip 标记为暂不可用。

## 关联

上游:sgl-deep-ep 0.1.1、DeepEP v2、DeepEP PR #10(SBO 支持待修)。同区域:DeepSeek/DeepEP 相关 PR(#23882 rebase 中也涉及 DeepEP PD 拓扑)、PR #48468 等 NVIDIA 依赖升级。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
