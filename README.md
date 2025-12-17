# vv-asr

语音识别工具，基于 FunASR 框架，支持多种 ASR 模型。

## 特性

- 支持多种模型：SenseVoiceSmall、Fun-ASR-Nano 等
- 自动设备检测：CUDA / MPS / CPU
- 自动半精度优化：bfloat16 / float16
- VAD 语音活动检测
- 热词增强识别
- 批量处理多个音频文件
- 输出带时间戳的转录结果

## 安装

```bash
uv sync
```

## 使用

```bash
# 基本用法
uv run python asr_tool.py audio.mp3

# 批量识别
uv run python asr_tool.py *.mp3 -o ./results

# 指定语言
uv run python asr_tool.py audio.mp3 -l zh

# 添加热词提高识别准确度
uv run python asr_tool.py audio.mp3 --hotwords 人工智能 机器学习

# 使用其他模型
uv run python asr_tool.py audio.mp3 -m FunAudioLLM/Fun-ASR-Nano-2512
```

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型名称 | iic/SenseVoiceSmall |
| `-d, --device` | 计算设备 (auto/cuda/mps/cpu) | auto |
| `-l, --language` | 语言 (auto/zh/en/ja) | auto |
| `-o, --output-dir` | 输出目录 | ./output |
| `-b, --batch-size` | 批处理大小 | 1 |
| `--hotwords` | 热词列表 | - |
| `--no-vad` | 禁用 VAD | - |
| `--vad-max-time` | VAD 最大分段时长 (ms) | 30000 |
| `--no-itn` | 禁用逆文本规范化 | - |
| `--no-compile` | 禁用 torch.compile 优化 | - |
| `--no-half` | 禁用半精度 | - |
| `--threads` | CPU 线程数 | 全部核心 |

## 输出格式

转录结果会保存为带时间戳的文本文件：

```
[00:00:00.000 --> 00:00:03.500] 第一句话
[00:00:03.500 --> 00:00:07.200] 第二句话
```

## 支持的模型

- `iic/SenseVoiceSmall` - 默认模型，速度快，效果好
- `FunAudioLLM/Fun-ASR-Nano-2512` - 更小的模型，需要 Qwen3-0.6B

## License

MIT
