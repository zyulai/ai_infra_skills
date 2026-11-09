# PR #36592: [Diffusion][Kernel] Fuse Wan FFN GELU epilogue

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-27 |
| 分类 | Attention / Kernel |
| Label | documentation, run-ci, diffusion, mergeable |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 wanvideo.py、新增 test_wan_gelu_mlp.py 并更新 skill 文档 |
| 链接 | https://github.com/sgl-project/sglang/pull/36592 |

## 一句话结论

把 Wan FFN 的第一投影与 GELU 合并进 cublasLt linear+GELU epilogue,五步去噪 profile 中独立 GELU 从 401 次/159ms 降到 12 次/0.03ms,端到端提速 1.85%(quality=high 可选)。

## 动机

Wan 的 feed-forward block 把第一投影与 `gelu_pytorch_tanh` 作为两个独立 kernel 执行。在 H100 上对 Wan2.1-T2V-1.3B 做五步去噪 profile,独立 GELU 占 401 次 launch、159.064 ms。仓库中已有 quality 门控的 cublasLt linear+GELU epilogue,复用它即可消除这条独立 GELU 链,同时保持 `quality=lossless` 的算子序列与输出完全不变。

## 优化点

为标准与 VSA transformer block 新增 Wan 专用 `_WanGELUMLP`(`act_type="gelu_pytorch_tanh"`),用 `mark_fused_gelu_site` 把其 `fc_in` 投影标记为 fused-linear-GELU 站点;forward 中当 `fused_gelu_active` 与 `can_use_linear_gelu`(quality gate + shape/backend guard)都通过时,直接调用 `fused_linear_gelu_tanh(x, weight, bias)` 一步完成投影+GELU(cublasLt 走 `gelubias` epilogue),否则保留原 `fc_in -> gelu_pytorch_tanh -> fc_out` 回退路径,`fc_out` 不变。`quality=lossless` 下该站点不挂载,输出保持 bit-exact。单元测试覆盖未挂载、挂载(atol/rtol=2e-2)与卸载后恢复 bit-exact 三条路径。另更新 diffusion performance skill 文档,注明 `quality=high` 会挂载 Wan FFN cublasLt GELU epilogue 与 Wan VAE RMSNorm+SiLU fast path。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py: 新增 _WanGELUMLP 并替换标准/VSA block 的 ffn
python/sglang/multimodal_gen/test/unit/test_wan_gelu_mlp.py: 挂载/卸载/恢复路径单元测试
python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance/SKILL.md: Wan quality=high 路径文档
```

核心 dispatch 逻辑:

```python
def forward(self, x):
    if fused_gelu_active(self) and can_use_linear_gelu(self.fc_in, x):
        x = fused_linear_gelu_tanh(x, self.fc_in.weight, self.fc_in.bias)
    else:
        x, _ = self.fc_in(x)
        x = self.act(x)
    x, _ = self.fc_out(x)
    return x
```

## 性能声明

主 workload(Wan2.1-T2V-1.3B,1xH100 80GB,832x480、81 帧、50 步去噪、guidance 3、seed 42,quality=high 对比),同 GPU A-B-B-A、每 process 一次 warmed request:baseline mean 去噪 45,237.27 ms / E2E 47,064.64 ms,fused mean 去噪 44,364.31 ms / E2E 46,192.15 ms,即 **1.93% denoising** 与 **1.85% end-to-end** 提速。五步 GPU trace:GPU kernel time 5,859.373 → 5,720.200 ms,launches 14,268 → 13,911,独立 GELU 159.064 ms/401 次 → 0.030 ms/12 次。

姊妹验证(FastVideo/FastWan2.2-TI2V-5B-FullAttn,1280x720x81,三步 WanDMDPipeline):对 main 5adc2880f9,A-B-B-A 均值去噪 1,374.930 → 1,352.524 ms(1.63%)、E2E 7,106.139 → 7,070.196 ms(0.51%);独立 BF16 GELU 从 90 次/32.571 ms 降到 2 次/0.007 ms,GEMM 使用 cublasLt `gelubias` epilogue,总 GPU kernel time 1,330.489 → 1,310.310 ms(1.52%)。

## 验证与回归风险

新增单元测试 `test_wan_gelu_mlp.py`(挂载后 atol/rtol=2e-2 一致、卸载后 `torch.equal` bit-exact),与 `test_sites.py` 合计 23 passed;rebase 到官方 main a3ae667d67 后同 23 个测试在 H100 复跑通过,官方 targeted CI run 33034992448 通过。输出质量:对未融合 quality=high 基线,首/中/尾关键帧 SSIM 0.960655/0.954638/0.947188,PSNR 36.12/34.32/33.75 dB(FastWan2.2 为 SSIM 0.938424/0.928753/0.931103、PSNR 32.16/30.47/30.27 dB)。风险点:该优化只在 `quality=high` 生效,lossless 路径不变;输出为 close contract 而非 bit-exact,多步去噪的舍入差异需依赖 SSIM/PSNR 验收;Wan 不支持 BCG,无 BCG 性能声明;AMD ROCm 7.2 CI 显示失败标记(x),正文未说明。

## 关联

同区域优化:diffusion kernel 体系中的 `fused_linear_gelu_tanh`(FLUX.1/GLM-Image 等已用的 linear+GELU 站点机制)、`mark_fused_gelu_site`/`BitExactFusionGate` 质量门控框架;同批 PR #38044(LingBot MoE routing 融合)同为 diffusion kernel 系列。PR 正文未给出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
