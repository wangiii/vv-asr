#!/usr/bin/env python3
"""
ASR工具 - 基于FunAudioLLM/Fun-ASR模型进行语音识别
"""

import argparse
import os
import sys
from pathlib import Path


def get_best_device() -> str:
    """自动检测最佳计算设备"""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3/M4) GPU
        return "mps"
    else:
        return "cpu"


def transcribe_audio(
    audio_path: str,
    model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512",
    device: str = "cuda:0",
    language: str = "auto",
    hotwords: list = None,
    use_vad: bool = True,
    vad_max_segment_time: int = 30000,
    itn: bool = True,
    batch_size: int = 1,
) -> str:
    """
    对音频文件进行ASR识别

    Args:
        audio_path: 音频文件路径
        model_name: 模型名称，默认为 FunAudioLLM/Fun-ASR-Nano-2512
        device: 计算设备，如 "cuda:0" 或 "cpu"
        language: 语言，可选 "auto", "zh", "en", "ja"
        hotwords: 热词列表，用于增强特定词汇的识别
        use_vad: 是否使用VAD（语音活动检测）进行长音频分段
        vad_max_segment_time: VAD最大分段时间（毫秒）
        itn: 是否启用逆文本规范化（如将"两千零二十三"转为"2023"）
        batch_size: 批处理大小

    Returns:
        识别出的文本
    """
    from funasr import AutoModel

    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 构建模型参数
    # 如果是本地路径，使用对应的 model.py；否则尝试自动解析
    if os.path.isdir(model_name):
        remote_code = os.path.join(model_name, "model.py")
    else:
        # 对于远程模型，使用默认缓存路径
        cache_dir = os.path.expanduser(f"~/.cache/modelscope/hub/models/{model_name}")
        remote_code = os.path.join(cache_dir, "model.py") if os.path.exists(cache_dir) else "./model.py"

    model_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "remote_code": remote_code,
        "device": device,
        "disable_update": True,  # 禁用更新检查，加快启动速度
    }

    # 如果启用VAD
    if use_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": vad_max_segment_time}

    # 初始化模型
    print(f"正在加载模型: {model_name}")
    print(f"使用设备: {device}")
    model = AutoModel(**model_kwargs)

    # 构建生成参数
    generate_kwargs = {
        "input": [audio_path],
        "cache": {},
        "batch_size": batch_size,
        "language": language,
        "itn": itn,
    }

    if hotwords:
        generate_kwargs["hotwords"] = hotwords

    # 执行识别
    print(f"正在识别音频: {audio_path}")
    result = model.generate(**generate_kwargs)

    # 提取文本
    text = result[0]["text"]
    return text


def transcribe_multiple(
    audio_paths: list,
    output_dir: str = None,
    **kwargs
) -> dict:
    """
    批量识别多个音频文件

    Args:
        audio_paths: 音频文件路径列表
        output_dir: 输出目录，如果指定则保存结果到txt文件
        **kwargs: 传递给transcribe_audio的其他参数

    Returns:
        字典，键为音频路径，值为识别文本
    """
    results = {}

    for audio_path in audio_paths:
        try:
            text = transcribe_audio(audio_path, **kwargs)
            results[audio_path] = text
            print(f"\n[{audio_path}]")
            print(f"识别结果: {text}")

            # 保存到文件
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = Path(audio_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"结果已保存到: {output_path}")

        except Exception as e:
            print(f"识别失败 [{audio_path}]: {e}")
            results[audio_path] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ASR工具 - 基于FunAudioLLM/Fun-ASR模型进行语音识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 识别单个音频文件
  python asr_tool.py audio.mp3

  # 识别多个音频文件
  python asr_tool.py audio1.mp3 audio2.wav audio3.flac

  # 使用CPU进行识别
  python asr_tool.py audio.mp3 --device cpu

  # 指定语言为中文
  python asr_tool.py audio.mp3 --language zh

  # 添加热词增强识别
  python asr_tool.py audio.mp3 --hotwords "人工智能" "机器学习"

  # 保存结果到指定目录
  python asr_tool.py audio.mp3 --output-dir ./results

  # 禁用VAD（适用于短音频）
  python asr_tool.py audio.mp3 --no-vad
        """
    )

    parser.add_argument(
        "audio_files",
        nargs="+",
        help="要识别的音频文件路径（支持多个文件）"
    )
    parser.add_argument(
        "--model", "-m",
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="模型名称 (默认: FunAudioLLM/Fun-ASR-Nano-2512)"
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="计算设备 (默认: auto自动检测，可选: cuda:0, mps, cpu)"
    )
    parser.add_argument(
        "--language", "-l",
        default="auto",
        choices=["auto", "zh", "en", "ja"],
        help="语言 (默认: auto)"
    )
    parser.add_argument(
        "--hotwords",
        nargs="*",
        help="热词列表，用于增强特定词汇的识别"
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="禁用VAD（语音活动检测）"
    )
    parser.add_argument(
        "--vad-max-time",
        type=int,
        default=30000,
        help="VAD最大分段时间（毫秒，默认: 30000）"
    )
    parser.add_argument(
        "--no-itn",
        action="store_true",
        help="禁用逆文本规范化"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="输出目录，保存识别结果到txt文件"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="批处理大小 (默认: 1)"
    )

    args = parser.parse_args()

    # 自动检测设备
    device = args.device if args.device != "auto" else get_best_device()
    print(f"使用计算设备: {device}")

    # 执行识别
    results = transcribe_multiple(
        audio_paths=args.audio_files,
        output_dir=args.output_dir,
        model_name=args.model,
        device=device,
        language=args.language,
        hotwords=args.hotwords,
        use_vad=not args.no_vad,
        vad_max_segment_time=args.vad_max_time,
        itn=not args.no_itn,
        batch_size=args.batch_size,
    )

    # 打印汇总
    print("\n" + "=" * 50)
    print("识别完成")
    print(f"成功: {sum(1 for v in results.values() if v is not None)}")
    print(f"失败: {sum(1 for v in results.values() if v is None)}")


if __name__ == "__main__":
    main()
