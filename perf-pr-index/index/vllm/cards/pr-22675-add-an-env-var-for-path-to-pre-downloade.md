# PR #22675: add an env var for path to pre-downloaded flashinfer cubin files

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @842974287 |
| 合并 | 2025-08-22 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/22675 |

## 一句话结论

新增 `VLLM_HAS_FLASHINFER_CUBIN` 环境变量,声明 cubin 已预下载后跳过 NVIDIA artifactory 网络检查,避免离线/受限环境误降级 trtllm attention 后端。

## 动机

前序 PR #21893 为 vLLM 增加了到 NVIDIA cubin artifact 端点的网络可达性检查(带 5 秒超时的 `requests.get`),检查失败时会从 trtllm attention 后端退出。对已经把 cubin 文件预下载到本地(经 `FLASHINFER_CUBIN_DIR` 指定路径、flashinfer 直接读取)的环境,这个联网检查既带来无谓的启动阻塞,又在没有外网的情况下错误地禁用了 trtllm 后端,需要给用户一个显式声明"Cubin 已就绪"的出口。

## 优化点

机制极小:`vllm/envs.py` 注册布尔型环境变量 `VLLM_HAS_FLASHINFER_CUBIN`(语义为"已预下载 cubin 文件,flashinfer 将直接读取");`vllm/utils/flashinfer.py` 的 `has_nvidia_artifactory()` 在函数开头检查该变量,若已设置则直接返回 True,跳过对 `FLASHINFER_CUBINS_REPOSITORY` 的网络请求。cubin 路径本身仍由 flashinfer 侧的 `FLASHINFER_CUBIN_DIR` 承担,本 PR 只负责让 vLLM 的后端可用性判定信任该声明。

## 关键改动

```text
vllm/envs.py: 新增 VLLM_HAS_FLASHINFER_CUBIN 环境变量
vllm/utils/flashinfer.py: has_nvidia_artifactory() 优先信任该变量,直接返回 True
```

核心片段:

```python
# Since FLASHINFER_CUBIN_DIR defines the pre-downloaded cubins path, when
# it's true, we could assume the cubins are available.
if envs.VLLM_HAS_FLASHINFER_CUBIN:
    return True
```

## 性能声明

PR 正文未给出性能数字,收益是定性的:离线或网络受限环境不再支付 5 秒超时的连通性探测(且探测失败导致的 trtllm attention 后端禁用被消除),已预下载 cubin 的部署可以直接走 trtllm 路径。PR 未注明测试条件。

## 验证与回归风险

diff 未包含任何测试;改动为一个 env 读取与一个短路返回。风险面:变量置真时 vLLM 不再验证 cubin 是否真的存在,若用户声明与实际不符,后续 flashinfer 加载 cubin 时才会失败,错误出现的位置更靠后;不影响未设置该变量的默认行为(仍做网络检查)。

## 关联

前序:#21893(引入网络检查);同区域:pr-24197(envs 中同步出现 `VLLM_HAS_FLASHINFER_CUBIN`)、pr-19118(FlashInfer 后端默认选择)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
