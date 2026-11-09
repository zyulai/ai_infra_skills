# PR #27297: [diffusion] Optimize LingBot realtime transport and camera conditioning

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-06-05 |
| 分类 | Other(tier: major) |
| Label | run-ci, diffusion, run-ci-extra |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 realtime 输出 adapter、realtime API、lingbot_world 模型、causal denoising stage 与测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/27297 |

## 一句话结论

LingBot realtime:缓存 camera conditioner scale/shift、raw RGB 改无损原始字节传输,steady chunk 总时延 -10.3%。

## 动机

LingBot realtime serving 有两处延迟:(1) 每个 DiT block 的 camera conditioner 都对同一 `c2ws_plucker_emb` 重复计算 scale/shift 线性投影;(2) 默认 realtime raw RGB 输出走 delta-gzip 压缩,每 chunk 的 payload 构建约 20-22 ms,且整个 msgpack header+payload 打包后一次发送。

## 优化点

第一,条件缓存:`LingBotWorldCamConditioner` 把 scale/shift 计算拆成 `compute_scale_shift`;transformer 在 forward 开始处一次预取所有 block 的 scale/shift(逐 block cache,entry key 为 conditioner id),block 直接接收 `cam_conditioner_scale_shift`。缓存挂在 forward_batch 的 request cache(`lingbot_cam_conditioner`),source key 由 data_ptr/shape/stride/dtype/device/_version 组成,源张量变化(含 `_version` 自增)即失效。仅当 `enable_sequence_shard` 且 Ulysses 并行 world size > 1 且处于正常 denoise timestep(current_timestep >= 0)时启用;context-cache 更新 forward 仍重算,保证因果状态更新 bitwise 一致。第二,传输:raw RGB 默认改为无损原始字节(删除 delta-gzip 分支与 reference frame 状态);大 payload 超过 `FRAME_BATCH_PACK_OFFLOAD_BYTES` 时改为小 msgpack header 与 raw bytes 两次 `send_bytes`,collector 兼容两种消息;websocket `/generate` 在 accept 后等待 `server_warmup_done` 再接受生成请求(CI OOM 修复)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py: compute_scale_shift 拆分、逐 block scale/shift 缓存与预取
python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_output_adapter.py: raw 直传、header/payload 分离发送
python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py: websocket 等待 server warmup
python/sglang/multimodal_gen/runtime/pipelines_core/.../lingbot_world_causal_denoising.py: 每步 pop 旧 cam conditioner 缓存
```

## 性能声明

4x H200 LingBot realtime,832x480,6 chunks,以 chunks 3/4/5 为 steady-state:

| Variant | steady chunk total | steady scheduler forward | raw payload build |
| --- | ---: | ---: | ---: |
| main baseline | 2360 ms | 2252 ms | 约 20-22 ms/chunk |
| 本 PR | 2114 ms | 2092 ms | 约 1 ms/chunk |

即 10.3% steady chunk total 改善、7.1% scheduler-forward 改善。一致性:4xH200 realtime 输出与 baseline bitwise 一致(frames `[45, 480, 832, 3]`,sha256 `e7692f77...1d8d`)。

## 验证与回归风险

`test_realtime_output_transport.py` 与 `test_lingbot_causal_denoising.py` 共 27 个用例通过(最新 CI-fix 验证),新增覆盖:scale/shift 直通等价、缓存命中/源张量失效、非 sequence-shard 与单 Ulysses world 跳过缓存、context update 跳过缓存、header+payload 分离消息。风险:缓存正确性依赖 source key 覆盖所有变化途径(data_ptr/_version 等);delta-gzip 编码被删除,旧客户端不再得到 `delta_reference` 语义;CI 单卡 OOM 案例通过禁用 LingBot realtime warmup 解决(启动行为变化)。

## 关联

同属 multimodal_gen diffusion realtime 线;其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
