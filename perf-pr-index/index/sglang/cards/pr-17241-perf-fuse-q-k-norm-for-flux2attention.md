# PR #17241: [Perf] fuse q, k norm for Flux2Attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zminglei |
| 合并 | 2026-01-19 |
| 分类 | Attention / Kernel (major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 2 个文件:apply_qk_norm 修复与 Flux2 接入 |
| 链接 | https://github.com/sgl-project/sglang/pull/17241 |

## 一句话结论

让 Flux2Attention 的 Q/K RMSNorm 走已有的融合 qk-norm kernel,FLUX.2-dev 端到端生成 24.32s→23.76s(约 2.5%)。

## 动机

Flux2Attention 的 forward 里 query/key 分别调用 `norm_q`/`norm_k` 两个独立 RMSNorm,各起一次 kernel;multimodal_gen 的 `apply_qk_norm` 已有融合 QK-norm 的 CUDA 快路径,但 Flux2 没有接上,且 `unflatten` 之后张量可能非连续,直接融合会引入隐式拷贝或形状问题。

## 优化点

两处改动。其一,`apply_qk_norm` 在进入融合路径前显式检查 `q`/`k` 的 contiguity,非连续时先 `.contiguous()`,融合 in-place kernel 跑完后按原 shape `view` 返回(此前 shape 变量只在 fallback 分支里取);融合不可用时(AMD/ROCm)保持原有的分别 RMSNorm fallback 并打 warning。其二,`flux_2.py` 的两个 attention forward(主分支与 added_kv_proj 分支)把 `self.norm_q(query)` + `self.norm_k(key)` 两行替换为一次 `apply_qk_norm(query, key, self.norm_q, self.norm_k, self.head_dim)`,把两次独立 norm launch 合并为一次融合 kernel 调用。机制上是消除每层 transformer 的两个小 kernel launch,属于 launch/访存开销削减,不改变计算语义。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/layernorm.py: apply_qk_norm 增加 contiguous 保证与 shape 还原,使融合 CUDA 路径可靠生效
python/sglang/multimodal_gen/runtime/models/dits/flux_2.py: 两个 attention forward 改用 apply_qk_norm 融合调用
```

接入改动:

```python
-        query = self.norm_q(query)
-        key = self.norm_k(key)
+        query, key = apply_qk_norm(query, key, self.norm_q, self.norm_k, self.head_dim)
```

## 性能声明

测试条件:`black-forest-labs/FLUX.2-dev`,`sglang generate` 单 prompt("A Logo With Bold Large Text: SGL Diffusion")单图输出;PR 未注明 GPU 型号与精度设置。

- 端到端(Pixel data / batch 处理总时长):24.32 s → 23.76 s,约 **2.5% e2e latency improvement**。
- DenoisingStage:平均每步 0.4453 s → 0.4346 s,总计 22.6614 s → 22.1209 s。
- DecodingStage 与 Peak GPU memory(64.49 GB)不变。

## 验证与回归风险

无新增测试(checklist 全部未勾选);验证为上述单次生成的 A/B 日志对比,无准确率数据。风险面:改动只影响 multimodal_gen 的 `apply_qk_norm` 与 Flux2 模型路径;contiguous 补齐可能在非连续输入时引入一次额外拷贝(换取融合 kernel 可用);AMD/ROCm 走原 fallback 不受影响。单样本 2.5% 的测量未说明重复次数,存在 run-to-run 噪声风险。

## 关联

同区域:multimodal_gen diffusion kernel 融合系列(如 #21440 qknorm+rope 融合、#29281 Conv3D cat-pad,pack 未确认相互关联);FLUX.2 模型接入相关 PR(pack 未列编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
