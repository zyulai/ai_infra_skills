# PR #58761: [ROCm][CI] Make ROCm 10.0 the default image; keep ROCm 7.2 as -rocm72

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Rohan138 |
| 合并 | 2026-10-18 |
| 分类 | Other(tier: major) |
| Label | documentation, rocm, ready, torch.compile, ci/build, pooling |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 19 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/58761 |

## 一句话结论

CI/发布切换:ROCm 10.0(TheRock)成为默认镜像与 wheel,ROCm 7.2 过渡期保留为 `-rocm72`。

## 动机

vLLM 的 ROCm 支持迁移到 TheRock(ROCm 10.0)栈,需要默认镜像、wheel 索引与发布流水线整体切换,同时给用户保留 ROCm 7.2 过渡通道,避免一刀切断供。

## 优化点

Dockerfile 改名互换:`Dockerfile.rocm{,_base}` → `Dockerfile.rocm_72{,_base}`,`Dockerfile.rock{,_base}` → `Dockerfile.rocm{,_base}`,全部 CI 转为构建 ROCm 10.0;`:nightly`/`:latest` 为 ROCm 10.0(另标 `-rocm100`),7.2 发布为 `-rocm72`;`wheels.vllm.ai/rocm/` 改服务 ROCm 10.0(7.2 保留在 `rocm/<ver>/rocm723`),索引托管 vLLM/triton/FA/aiter/amdsmi 及重打包的 torch,并外链 AMD whl-next 的 ROCm SDK 与 GPU kernels。两套栈共用同一组脚本与流水线步骤:每个 ROCm 脚本以栈的 base Dockerfile 为参数,`.buildkite/scripts/rocm/stack.sh` 由此推导版本、variant、cache key 与镜像 tag;新增 `therock_wheels.py` 处理 TheRock wheel 适配(torch 的 triton pin 重写、外链生成、amdsmi loader patch);rocm.py 对 RCCL 2.30 设 `NCCL_RAS_ENABLE=0` 规避退出时 abort。

## 关键改动

```text
docker/Dockerfile.rocm{,_base} 与 Dockerfile.rocm_72{,_base}: 改名互换
.buildkite/scripts/rocm/stack.sh 与 .buildkite/scripts/cache-rocm-base-wheels.sh: 按 base Dockerfile 派生 variant/cache key
tools/vllm-rocm/therock_wheels.py(新): repack-torch / external-links / patch-amdsmi
vllm/platforms/rocm.py: NCCL_RAS_ENABLE=0 workaround
.buildkite/release-pipeline.yaml: 双栈 release 流水线与 perf-eval(ROCm -rocm100 镜像)触发
```

## 性能声明

PR 未提供性能数据(发布/CI 基础设施改动)。测试结果:YAML/shellcheck/ruff、`tests/tools/test_docker_build_metadata_args.py`、对假 S3 的发布/上传脚本 dry run、`uv pip compile "vllm[device-gfx942]"` 本地索引解析全部通过;完整镜像构建与 AMD 测试套件尚待运行。附带:pooling 测试 `test_token_classification` 容差 atol 3.2e-2 → 6e-2。

## 验证与回归风险

已知缺口(正文声明):wheel 安装缺镜像构建期 ROCm patch;amdsmi wheel 仍需硬件检查;首次运行需重建各 base 镜像(内部 ECR tag 改为 variant 命名)。用户面风险:默认安装栈变为 ROCm 10.0,需要 7.2 的用户须显式改镜像 tag 或 wheel index 路径。

## 关联

同批 ROCm 区域:#47972(MI355 GSM8K CI)、#51052(MI355X MoRIIO 验证)、#53175(rocm label)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
