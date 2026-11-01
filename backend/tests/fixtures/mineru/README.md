# MinerU 云端真实响应摘录

录制于 2026-10-24，云端 `vlm` 返回 `_version_name=3.4.4`。

- `attention`：Attention Is All You Need v7，完整 PDF 15 页。
- `lora`：LoRA v2，完整 PDF 26 页。
- `geometry`：Attention 第 4 页旋转 90°、第 6 页 CropBox 设为 `[70,100,550,730]` 的两页副本。

`content_list.json` 是按类型选择的原始行摘录，未修改内容、bbox 或页索引。`layout.json` 仅保留供应商标识与真实页尺寸；`pdf_info.json` 保留本地核验的原 PDF 几何信息。二进制图片、Markdown、凭据、任务 ID 与签名地址不在摘录中。测试应允许图片缺失，并保留其图注与页级来源。

`provenance.json` 包含来源链接、PDF / ZIP SHA-256、摘录行在完整结果中的索引及完整解析统计。完整私有结果保存在工作区 `reports/m1-local/`，目录已忽略。更换云端格式时应重新录制并核对，不能仅修改预期值让测试通过。

使用说明与验收见仓库 `docs/M1.md`。这些样例用于解析协议回归，不是检索 / 问答质量黄金集。
