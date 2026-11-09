# PR #33932: Install DeepEP from release wheels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2026-08-07 |
| 分类 | MoE |
| Label | high priority, dependencies, deepseek, run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含大量文件(约 28 个),另有 14 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/33932 |

## 一句话结论

CUDA CI 从源码编译 DeepEP 改为安装发布的 `sgl-deep-ep==0.1.0` wheel,依赖安装时间从 4 分钟降到 30 秒,并撤掉专用 DeepEP runner 配置。

## 动机

SGLang 开始发布 `sgl-deep-ep` wheel,而 CUDA CI 仍在专用 runner 配置与测试套件里从源码重建 DeepEP(clone 仓库、检查特定 commit、按 CUDA 版本选择 TORCH_CUDA_ARCH_LIST 编译,还要为 Grace Blackwell 用 hybrid-ep 分支并 sed 修改超时常量)。源码构建慢、脆弱且需要独立的 `deepep-*` runner 拓扑。改用发布 wheel 可消除这套定制安装链。

## 优化点

安装路径重构:pyproject 增加 pin `sgl-deep-ep==0.1.0`;CUDA 13 runner 直接用 PyPI 上的公共 wheel,CUDA 12 runner 从 SGLang 的 cu129 wheel index 安装本地版本号(`sgl-deep-ep==<version>+<CU_VERSION>`,满足公共版本 pin,后续 editable 安装保留该 CUDA 匹配 wheel);GDRCopy 及其系统依赖(RDMA/infiniband 包)并入通用 CUDA 依赖安装脚本,仅在 >=4 GPU 的 runner 上构建(1/2 GPU job 跳过 DKMS 构建,已安装则复用),pip 安装前同时卸载 legacy `deep-ep` 与 `sgl-deep-ep` 源分发避免混合安装。CI 拓扑简化:删除 `ci_install_deepep.sh`(177 行)与 `FORCE_REBUILD_DEEPEP`、`GRACE_BLACKWELL` 等控制变量;`deepep-*` runner 配置全部移除,16 个注册测试合并进对应标准硬件套件(不改 base/extra stage 元数据);删除 6 个专用 workflow job;NPU 的 DeepEP 安装路径保持不变。

## 关键改动

```text
python/pyproject.toml: pin sgl-deep-ep==0.1.0
scripts/ci/cuda/ci_install_dependency.sh: install_gdrcopy(>=4 GPU)+ install_cuda12_deepep_wheel(cu129 index)
scripts/ci/cuda/ci_install_deepep.sh: 删除(源码构建器)
scripts/ci/runner_configs.yml: deepep-* 配置移除,4-gpu-gb300/8-gpu-h20 改用默认 install
.github/workflows/{pr-test,pr-test-extra,nightly-test-nvidia,...}.yml: 专用 DeepEP job 删除,套件合并
```

## 性能声明

- 依赖安装时间:**4 min -> 30 sec**(PR 正文原句 "Dependency Installation time: 4min -> 30sec")。
- 无模型/运行时改动,不涉及推理吞吐与精度(checklist 中两项标记 Not applicable:CI-only migration)。

## 验证与回归风险

验证:repository pre-commit hooks、DeepEP CI 迁移契约测试、34 个 CI model inventory 单测、全部 workflow YAML 与 pyproject 解析、base/extra workflow 分区计算(含 6 个目标 job)、Ruff E9/F63/F7/F82、PyPI CUDA 13 与 cu129 wheel 解析验证、16 个迁移注册的套件并集与 stage 元数据逐一核对。风险面:CUDA 12 runner 依赖 SGLang 私有 wheel index 的可用性;wheel 与源码构建版本之间可能存在行为差异(此前 Grace Blackwell 走 hybrid-ep 分支 + 特定 commit);GDRCopy 构建从 deepep 专用脚本迁到通用脚本,1-2 GPU runner 不再预装(DeepEP 测试仅在 >=4 GPU 上运行);`verify_imports` 增加 `deep_ep` 导入检查兜底。

## 关联

同区域:DeepSeek/DeepEP MoE 全量通信栈(labels 含 deepseek)。运行时性能无关联改动,区域内无其他已索引 PR 编号,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
