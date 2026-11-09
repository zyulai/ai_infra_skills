# PR #48064: [Distributed][Perf] Enable FlashInfer MNNVL allreduce RMS quant fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmangkad |
| 合并 | 2026-07-13 |
| 分类 | Quantization(major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/48064 |

## 一句话结论

把 AR+RMSNorm+量化融合(FP8/FP4)从 trtllm 后端开放给 MNNVL 后端,使多节点量化推理也能吃上 allreduce 融合。

## 动机

vLLM 的 allreduce+RMSNorm 融合 pass 支持 FP8/FP4 量化模式,但 quant fusion 的 workspace 此前写死只用 trtllm 后端,而 trtllm 只支持单节点——多节点部署直接禁用 quant fusion(打 warning 并返回 None)。当前 FlashInfer 的 MNNVL allreduce kernel 本身已支持量化:`trtllm_mnnvl_allreduce.cuh` 的 `QuantType` 枚举含 `kFP8=1` 与 `kFP4=2`。能力存在但 vLLM 侧没接上。

## 优化点

两处改动:(1) `flashinfer_all_reduce.py` 的 `get_fi_ar_quant_workspace` 不再写死 trtllm:backend 由 `_resolve_fi_ar_backend()` 按 `VLLM_FLASHINFER_ALLREDUCE_BACKEND` 解析,`auto` 优先 mnnvl、仅单节点且 mnnvl multicast 不可用时回退 trtllm;mnnvl workspace 创建失败(无 NVSwitch multicast)时警告并回退 trtllm;多节点 + 显式 trtllm 从"静默禁用"改为直接 raise ValueError,复用同 backend 的非 quant workspace。(2) `allreduce_rms_fusion.py`:layout_code 的适用后端从仅 trtllm 扩到 ("trtllm", "mnnvl")(vLLM quant pattern 一律用 SWIZZLED_128x4);NVFP4 两个 pattern 的 scale buffer 做 dtype view 适配——vLLM 侧 int32 打包的 `output_scale` 经 `aten.view.dtype` 以 FP8 视图传给 FlashInfer,返回后再 view 回 int32,避免拷贝。

## 关键改动

```text
vllm/distributed/device_communicators/flashinfer_all_reduce.py: quant workspace 后端解析与 mnnvl/单节点回退
vllm/compilation/passes/fusion/allreduce_rms_fusion.py: mnnvl 支持 SWIZZLED layout;NVFP4 scale dtype view 适配
```

关键片段(scale 适配):

```python
def _view_nvfp4_scale_out_for_flashinfer(scale_out):
    # View vLLM's packed NVFP4 scale buffer as FP8 for FlashInfer.
    return torch.ops.aten.view.dtype(scale_out, FP8_DTYPE)
```

## 性能声明

PR 正文未给出量化数字,仅附一张性能对比截图(素材无法读取其中数值)与指向 flashinfer 源码的 QuantType 枚举链接。收益定性为:多节点量化部署从"无融合"变为 AR+RMSNorm+FP8/FP4 量化单 kernel 融合。加速幅度 PR 未注明。

## 验证与回归风险

PR 正文未提供 Test Plan/Test Result 章节,也未注明新增自动化测试(素材正文极简)。风险面:quant fusion workspace 的后端选择与回退逻辑涉及多分支(显式 backend、auto、mnnvl 失败回退、复用已有 workspace),拓扑差异(单节点/多节点/无 multicast)下的行为需覆盖;NVFP4 的 dtype view 假设 int32 与 FP8 缓冲区字节数匹配(view 若形状/对齐不符会出错);多节点 + 显式 trtllm 现在会 raise,原先"静默禁用"的用户配置会变成启动报错。

## 关联

同区域:PR #46392(ws=16 fused allreduce 阈值标定,同一融合 pass)、PR #37756(SM10.3 AR tuning);上游能力位于 flashinfer trtllm_mnnvl_allreduce.cuh。PR 未注明其他前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
