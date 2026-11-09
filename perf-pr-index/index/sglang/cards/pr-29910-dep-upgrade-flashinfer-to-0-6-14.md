# PR #29910: [Dep] Upgrade flashinfer to 0.6.14

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2026-07-10 |
| 分类 | Attention / Kernel |
| Label | dependencies, run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/29910 |

## 一句话结论

将 FlashInfer 依赖从 0.6.12 升级到 0.6.14,并调整 flashinfer-cubin 的安装与分发方式。

## 动机

PR 正文为模板、未填写动机。从 diff 看,目的就是把 FlashInfer 全家桶(python/cubin/jit-cache)统一升到 0.6.14,使 SGLang 搭载 FlashInfer 较新版本的 attention/MoE/通信内核;同时 flashinfer-cubin 从 pyproject 固定依赖中移出,改为按 flashinfer_python 同版本显式安装。

## 优化点

版本对齐:Dockerfile 的 FLASHINFER_VERSION 从 0.6.12 到 0.6.14;pyproject 中 `flashinfer_python[cu13]==0.6.14`,并删去 `flashinfer_cubin==0.6.12` 固定依赖;engine.py 对 flashinfer attention backend 的 assert_pkg_version 提到 0.6.14。安装方式:flashinfer-cubin 是 CUDA 版本无关的包(与 jit-cache 不同),因此 Docker 与 CI 脚本都改用不带 cu 后缀的 `--index-url https://flashinfer.ai/whl` 安装,版本号取 flashinfer_python 所需版本;Docker 构建阶段把 flashinfer_cubin 一并 stage 进 /flashinfer_jit_output 并始终 COPY 进运行镜像(jit-cache 仍按需安装);CI 脚本新增 install_flashinfer_cubin 步骤。测试适配:test/registered/moe/test_cutedsl_moe.py 的 v2 CUDA-graph parity 测试改为真实 torch.cuda.CUDAGraph 捕获 + 两次 replay 对比,非图与图输出容差 1e-2,两次 replay 间最大差值阈值放宽到 0.5(原先要求两次 pass 结果逐位一致),对齐 0.6.14 下 CuteDSL MoE wrapper 在图内的行为。

## 关键改动

```text
python/pyproject.toml: flashinfer_python[cu13]==0.6.14,移除 flashinfer_cubin 固定依赖
docker/Dockerfile: FLASHINFER_VERSION=0.6.14,新增 flashinfer-cubin 安装与镜像分发
scripts/ci/cuda/ci_install_dependency.sh: install_flashinfer_cubin 步骤(版本跟随 flashinfer_python)
python/sglang/srt/entrypoints/engine.py: flashinfer_python 最低版本断言升至 0.6.14
test/registered/moe/test_cutedsl_moe.py: CuteDSL MoE v2 cuda-graph parity 测试改为真图捕获/replay
```

## 性能声明

PR 正文未提供任何性能或准确性数字(Motivation/Speed Tests 等模板段为空)。升级本身带来 FlashInfer 0.6.14 的内核更新,但该 PR 未给出对比测量。

## 验证与回归风险

验证手段为 CI(Extra 通过、Base no_entry_sign)。回归面:所有 flashinfer 依赖路径(attention backend、MoE runner、通信、sampling 等)随版本整体变化;flashinfer-cubin 的安装方式变化影响 Docker/CI 环境(不再经 pyproject 解析);CuteDSL MoE cuda-graph parity 测试的确定性要求被放宽(两次 replay 最大差值 <0.5),说明 0.6.14 下该路径图内 replay 存在被接受的数值波动,依赖逐位一致性的下游需注意。

## 关联

依赖链:此前版本为 0.6.12(升级 PR 编号素材未给出);早期 FlashInfer 集成升级如 PR #3550(0.2.1 MLA);flashinfer 通信/MoE 使用方如 #22394、#21637。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
