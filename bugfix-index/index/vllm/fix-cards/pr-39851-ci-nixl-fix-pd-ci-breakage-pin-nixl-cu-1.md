# PR #39851: [CI][NIXL] Fix PD CI breakage: pin nixl-cu{12,13} versions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39851](https://github.com/vllm-project/vllm/pull/39851) |
| 作者 | ZhanqiuHu |
| 合并日期 | 2026-04-15 |
| labels | ready, ci/build, kv-connector |
| 热度 | 👍 6 · 💬 3 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
新发布的 nixl-cu12==1.0.1 携带按 libcudart.so.12 编译的 nixl_ep,在 CUDA 13 CI 上 import 崩溃;将 nixl-cu12/cu13 后端包一并钉在 < 0.10.0。

## 问题现象
PD(disaggregated prefill)CI 在 CUDA 13 runner 上因 `nixl_ep` import 崩溃而挂掉。触发条件:`nixl-cu12==1.0.1` 于当日 19:38 UTC 新发布到 PyPI,依赖解析装上了它。正文给出的 pip 装包记录(照抄):

```
[2026-04-14T21:21:43Z]  + nixl==0.9.0
[2026-04-14T21:21:43Z]  + nixl-cu12==1.0.1
[2026-04-14T21:21:43Z]  + nixl-cu13==1.0.1
```

对照组:buildkite build 61159(`nixl-cu12==1.0.0`,无 `nixl_ep`)通过;buildkite build 61308(`nixl-cu12==1.0.1`,`nixl_ep` import 崩溃)失败。影响面为使用 NIXL kv-connector 的 CI/环境,不涉及模型代码。

## 根因
正文披露:`nixl-cu12==1.0.1` 附带的 `nixl_ep` 是按 `libcudart.so.12` 编译的,在 CUDA 13 CI runner 上直接崩溃。而仓库里原有的 `nixl < 0.10.0` 约束只钉住 meta-package(`nixl==0.9.0`),并不约束后端包 `nixl-cu12`/`nixl-cu13`,因此新发布 1.0.1 绕过了版本约束被装进环境。

## 修复方案
临时修复:在 `requirements/kv_connectors.txt` 中显式钉住两个后端包,与 meta-package 同样约束 `< 0.10.0`:

```diff
 nixl[cu13] >= 0.7.1, < 0.10.0 # Required for disaggregated prefill
+nixl-cu12 >= 0.7.1, < 0.10.0
+nixl-cu13 >= 0.7.1, < 0.10.0
 mooncake-transfer-engine >= 0.3.8
```

只改依赖清单一行文件,不动任何代码。正文说明正式的版本升级修复由 @NickLucche 在 #39797 进行,本 PR 是临时钉版。

## 影响与回归风险
- 影响窗口:坏版本为 PyPI 2026-04-14 发布的 `nixl-cu12==1.0.1`/`nixl-cu13==1.0.1`;此前可用的对照是 `nixl-cu12==1.0.0`。正文未给出受影响的 vllm 提交范围。
- 回归面:仅影响 KV connector 依赖解析路径(`requirements/kv_connectors.txt`),强制后端包落在 0.7.1–0.10.0 区间;后续正式升级 nixl 时需同步解除或调整这两行 pin,否则会锁死后端版本。

## 关联
- [#39797](https://github.com/vllm-project/vllm/pull/39797)(正式版本升级,进行中)
- [#39521](https://github.com/vllm-project/vllm/pull/39521)(tracking)
- [buildkite build 61159(通过对照)](https://buildkite.com/vllm/ci/builds/61159)
- [buildkite build 61308(失败现场)](https://buildkite.com/vllm/ci/builds/61308)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
