#!/usr/bin/env python3
"""
vv-asr - 语音识别工具

基于 FunASR 框架的语音识别命令行工具，支持：
- 多种 ASR 模型 (SenseVoiceSmall, Fun-ASR-Nano 等)
- VAD 语音活动检测，自动分段并保留时间戳
- 自动设备检测 (CUDA/MPS/CPU) 和半精度优化
- 批量处理多个音频文件

核心处理流程:
1. 加载模型: build_model() 分别加载 VAD 和 ASR 模型
2. 音频识别: transcribe() 先用 VAD 切分音频，再对每段进行 ASR
3. 结果保存: save_result() 输出带时间戳的字幕格式

使用示例:
    uv run python asr_tool.py audio.mp3 -o ./output
"""

# ============================================================
# 初始化配置 - 在导入其他库前设置，避免不必要的警告输出
# ============================================================
import logging
import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*[Aa]ttention mask.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================
# 标准库和第三方库导入
# ============================================================
import argparse
import re
import shutil
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import torch

# ============================================================
# 全局配置常量
# ============================================================
DEFAULT_MODEL = "iic/SenseVoiceSmall"  # 默认 ASR 模型
DEFAULT_VAD_MAX_TIME = 30000  # VAD 最大分段时长 (毫秒)
DEFAULT_BATCH_SIZE = 1

# SenseVoice 输出格式: <|zh|><|NEUTRAL|><|Speech|><|woitn|>文本内容
# 需要用正则表达式清理这些标签
_LANG_TAG_RE = re.compile(r"<\|(?:zh|en|ja|yue|ko)\|>")  # 语言标签，用于分割句子
_OTHER_TAG_RE = re.compile(r"<\|[^|]+\|>")  # 其他标签 (情感、语音类型等)


# ============================================================
# 设备和数据类型检测
# ============================================================

@lru_cache(maxsize=1)
def get_device() -> str:
    """
    自动检测最佳计算设备

    优先级: CUDA GPU > Apple MPS > CPU
    使用 lru_cache 缓存结果，避免重复检测
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """
    根据设备选择最佳数据类型

    - CUDA: 优先 bfloat16 (更好的数值稳定性)，否则 float16
    - MPS/CPU: 使用 float32 (MPS 对半精度支持不完善)
    """
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# ============================================================
# 工具函数
# ============================================================

def format_timestamp(ms: int) -> str:
    """
    毫秒时间戳转换为 SRT 字幕格式: HH:MM:SS.mmm

    例: 65500 -> "00:01:05.500"
    """
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_qwen_model(model_name: str):
    """
    确保 Fun-ASR-Nano 所需的 Qwen3-0.6B 模型存在

    Fun-ASR-Nano 模型依赖 Qwen LLM 进行文本后处理，
    但 ModelScope 缓存目录结构与 HuggingFace 不同，
    需要手动下载并复制模型文件
    """
    cache_dir = Path.home() / f".cache/modelscope/hub/models/{model_name}"
    qwen_dir = cache_dir / "Qwen3-0.6B"
    if qwen_dir.is_dir() and not (qwen_dir / "model.safetensors").exists():
        print("正在下载 Qwen3-0.6B 模型...")
        from huggingface_hub import snapshot_download
        hf_path = Path(snapshot_download(repo_id="Qwen/Qwen3-0.6B"))
        for f in hf_path.iterdir():
            if f.is_file() and not (qwen_dir / f.name).exists():
                shutil.copy2(f, qwen_dir / f.name)


def clean_text(text: str) -> list[str]:
    """
    清理 SenseVoice 模型输出的标签，提取纯文本

    SenseVoice 输出格式示例:
        "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好世界<|zh|><|HAPPY|><|Speech|>很高兴"

    处理步骤:
    1. 用语言标签 <|zh|> 等分割文本 (每个语言标签代表一个句子边界)
    2. 移除每段中的其他标签 (情感、语音类型等)
    3. 过滤空白内容

    返回: 清理后的句子列表 ["你好世界", "很高兴"]
    """
    return [
        clean for part in _LANG_TAG_RE.split(text)
        if (clean := _OTHER_TAG_RE.sub("", part).strip())
    ]


def resample_audio(waveform, orig_sr: int, target_sr: int = 16000):
    """
    重采样音频到目标采样率

    SenseVoiceSmall 模型要求 16kHz 采样率输入。
    当音频采样率不匹配时 (如 48kHz)，直接传入会导致识别错误。

    Args:
        waveform: 音频波形数据 (numpy array)
        orig_sr: 原始采样率
        target_sr: 目标采样率，默认 16000Hz
    """
    if orig_sr == target_sr:
        return waveform
    import librosa
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def load_audio(audio_path: str, target_sr: int = 16000):
    """
    加载音频文件，支持多种格式 (m4a, mp4, webm 等)

    优先使用 soundfile 读取，如果格式不支持则通过 ffmpeg 转换。

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率，默认 16000Hz

    Returns:
        (waveform, sample_rate): 音频数据和采样率
    """
    import soundfile as sf
    import subprocess
    import tempfile
    import numpy as np

    # 首先尝试用 soundfile 直接读取
    try:
        waveform, sample_rate = sf.read(audio_path)
        if sample_rate != target_sr:
            waveform = resample_audio(waveform, sample_rate, target_sr)
        return waveform, target_sr
    except Exception:
        pass  # soundfile 不支持此格式，尝试 ffmpeg

    # 使用 ffmpeg 转换为 wav 格式
    # -i: 输入文件
    # -f wav: 输出格式
    # -ar: 采样率
    # -ac 1: 单声道
    # -acodec pcm_s16le: 16位 PCM 编码
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", str(target_sr), "-ac", "1", "-acodec", "pcm_s16le",
            "-loglevel", "error", tmp_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        waveform, sample_rate = sf.read(tmp_path)
        return waveform, sample_rate
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 转换失败: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError("需要安装 ffmpeg 来处理此音频格式") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ============================================================
# 模型构建
# ============================================================

def build_model(model_name: str, device: str, use_vad: bool, vad_max_time: int,
                use_compile: bool = True, use_half: bool = True):
    """
    构建 ASR 和 VAD 模型

    为什么分离 VAD 和 ASR 模型？
    --------------------------------
    FunASR 的 AutoModel 支持内置 VAD，但会丢失时间戳信息。
    分离加载后，我们可以：
    1. 先用 VAD 模型检测语音片段，获取每段的起止时间
    2. 再用 ASR 模型识别每个片段的文本
    3. 最终输出带时间戳的字幕

    Args:
        model_name: ASR 模型名称 (如 "iic/SenseVoiceSmall")
        device: 计算设备 ("cuda:0", "mps", "cpu")
        use_vad: 是否启用 VAD 语音活动检测
        vad_max_time: VAD 最大分段时长 (毫秒)
        use_compile: 是否启用 torch.compile 优化 (仅 CUDA)
        use_half: 是否使用半精度浮点数

    Returns:
        {"asr": ASR模型, "vad": VAD模型或None}
    """
    from funasr import AutoModel

    # Fun-ASR-Nano 需要额外的 Qwen 模型作为 LLM 后端
    if "Fun-ASR-Nano" in model_name:
        from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401 注册模型类
        ensure_qwen_model(model_name)

    dtype = get_dtype(device) if use_half else torch.float32
    dtype_name = str(dtype).split('.')[-1]
    print(f"加载模型: {model_name} (设备: {device}, 精度: {dtype_name})")

    # 分离加载 VAD 和 ASR 模型
    # VAD 模型: fsmn-vad (轻量级，用于检测语音活动区间)
    vad_model = None
    if use_vad:
        vad_model = AutoModel(model="fsmn-vad", device=device, disable_update=True)
        vad_model.vad_max_time = vad_max_time

    # ASR 模型: 语音识别主模型
    asr_model = AutoModel(model=model_name, device=device, disable_update=True)

    # 半精度优化: 减少显存占用，加速推理
    if use_half and dtype != torch.float32:
        try:
            asr_model.model = asr_model.model.to(dtype)
            print(f"已转换为 {dtype_name} 精度")
        except Exception as e:
            print(f"半精度转换失败: {e}")

    # torch.compile: PyTorch 2.0+ 的图编译优化
    # 注意: MPS 设备不支持 compile，首次运行需要编译时间
    if use_compile and hasattr(torch, "compile") and device != "mps":
        try:
            asr_model.model = torch.compile(asr_model.model, mode="default")
            print("已启用 torch.compile 优化 (首次运行需要编译)")
        except Exception as e:
            print(f"torch.compile 失败: {e}")

    return {"asr": asr_model, "vad": vad_model}


# ============================================================
# 核心转录逻辑
# ============================================================

def transcribe(models: dict, audio_path: str, **kwargs) -> dict:
    """
    识别单个音频文件，返回带时间戳的转录结果

    处理流程 (启用 VAD 时):
    --------------------------------
    1. 音频预处理: 加载音频文件 (支持 m4a/mp4/webm 等格式通过 ffmpeg)

    2. VAD 检测: 找出音频中所有语音片段的时间区间
       例: [[0, 5000], [7000, 12000]] 表示 0-5秒 和 7-12秒 有语音

    3. 分段识别: 对每个 VAD 片段:
       - 提取对应时间区间的音频数据
       - 调用 ASR 模型识别
       - 清理输出标签，提取纯文本
       - 记录文本和对应的时间戳

    4. 返回结果: {"text": 完整文本, "segments": [带时间戳的句子列表]}

    Args:
        models: build_model() 返回的模型字典
        audio_path: 音频文件路径
        **kwargs: 传递给 ASR 模型的额外参数 (language, batch_size 等)

    Returns:
        {
            "text": "完整识别文本",
            "segments": [
                {"text": "句子1", "start": 0, "end": 5000},
                {"text": "句子2", "start": 7000, "end": 12000},
            ]
        }
    """
    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    asr_model = models["asr"]
    vad_model = models.get("vad")

    # 预加载音频数据 (支持 m4a/mp4/webm 等格式)
    target_sr = 16000
    waveform, sample_rate = load_audio(audio_path, target_sr)

    segments = []  # 带时间戳的句子列表
    all_text = []  # 所有句子的纯文本

    with torch.inference_mode():  # 推理模式，禁用梯度计算
        if vad_model:
            # ========== VAD 模式: 分段识别，保留时间戳 ==========

            # Step 1: VAD 检测语音片段 (使用预加载的音频数据)
            vad_res = vad_model.generate(input=[waveform], cache={})[0]
            vad_segments = vad_res.get("value", [])  # [[start_ms, end_ms], ...]

            if vad_segments:
                # Step 2: 对每个 VAD 片段进行 ASR 识别
                for seg in vad_segments:
                    start_ms, end_ms = seg[0], seg[1]

                    # 时间戳 (毫秒) 转换为采样点索引
                    start_sample = int(start_ms * sample_rate / 1000)
                    end_sample = int(end_ms * sample_rate / 1000)

                    # 提取片段音频数据
                    segment_audio = waveform[start_sample:end_sample]
                    if len(segment_audio) == 0:
                        continue

                    # ASR 识别片段
                    res = asr_model.generate(input=[segment_audio], cache={}, **kwargs)[0]
                    text = res.get("text", "")
                    cleaned = clean_text(text)

                    # 记录每个句子及其时间戳
                    for sent in cleaned:
                        segments.append({"text": sent, "start": start_ms, "end": end_ms})
                        all_text.append(sent)
            else:
                # VAD 没有检测到语音 (可能是静音文件)
                res = asr_model.generate(input=[waveform], cache={}, **kwargs)[0]
                all_text = clean_text(res.get("text", ""))
        else:
            # ========== 非 VAD 模式: 直接识别整个文件 ==========
            res = asr_model.generate(input=[waveform], cache={}, **kwargs)[0]
            text = res.get("text", "")
            all_text = clean_text(text)

            # 尝试从模型结果中获取时间戳 (某些模型支持)
            if sentence_info := res.get("sentence_info"):
                segments = [
                    {"text": sent, "start": s["start"], "end": s["end"]}
                    for s in sentence_info
                    for sent in clean_text(s["text"])
                ]
            elif ts := res.get("timestamp"):
                segments = [{"text": sent, "start": ts[0][0], "end": ts[-1][1]} for sent in all_text]

    return {"text": "\n".join(all_text), "segments": segments}


# ============================================================
# 结果保存
# ============================================================

def save_result(result: dict, path: Path):
    """
    保存识别结果到文件

    输出格式 (有时间戳时):
        [00:00:00.630 --> 00:00:05.090] 第一句话
        [00:00:05.370 --> 00:00:10.200] 第二句话

    输出格式 (无时间戳时):
        纯文本内容
    """
    with path.open("w", encoding="utf-8") as f:
        if result["segments"]:
            lines = (
                f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text']}"
                for s in result["segments"]
            )
            f.write("\n".join(lines) + "\n")
        else:
            f.write(result["text"])


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="ASR工具 - 语音识别",
        epilog="示例: uv run python asr_tool.py audio.mp3 -o ./output")

    # 位置参数
    parser.add_argument("files", nargs="+", type=Path, help="音频文件")

    # 模型和设备选项
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"ASR 模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("-d", "--device", default="auto",
                        help="计算设备: auto/cuda/mps/cpu (默认: auto)")
    parser.add_argument("-l", "--language", default="auto",
                        choices=["auto", "zh", "en", "ja"],
                        help="语言: auto/zh/en/ja (默认: auto)")

    # 输出选项
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./output"),
                        help="输出目录 (默认: ./output)")

    # 高级选项
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hotwords", nargs="*", help="热词列表，提高特定词汇识别率")
    parser.add_argument("--no-vad", action="store_true", help="禁用 VAD 分段")
    parser.add_argument("--vad-max-time", type=int, default=DEFAULT_VAD_MAX_TIME,
                        help=f"VAD 最大分段时长 ms (默认: {DEFAULT_VAD_MAX_TIME})")
    parser.add_argument("--no-itn", action="store_true", help="禁用逆文本规范化")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile 优化")
    parser.add_argument("--no-half", action="store_true", help="禁用半精度 (float16/bfloat16)")
    parser.add_argument("--threads", type=int, help="CPU 线程数 (默认: 全部核心)")

    args = parser.parse_args()

    # 设备和线程配置
    device = get_device() if args.device == "auto" else args.device
    num_threads = args.threads or os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"设备: {device}, CPU线程: {num_threads}")

    # 加载模型
    models = build_model(args.model, device, not args.no_vad, args.vad_max_time,
                         use_compile=not args.no_compile, use_half=not args.no_half)

    # ASR 生成参数
    gen_kwargs = {"batch_size": args.batch_size, "language": args.language, "itn": not args.no_itn}
    if args.hotwords:
        gen_kwargs["hotwords"] = args.hotwords

    # 批量处理音频文件
    success, failed = 0, 0
    for audio in args.files:
        try:
            result = transcribe(models, str(audio), **gen_kwargs)
            print(f"\n[{audio}] {result['text']}")

            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                out_path = args.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{audio.stem}.txt"
                save_result(result, out_path)
                print(f"保存: {out_path}")
            success += 1
        except Exception as e:
            print(f"失败 [{audio}]: {e}")
            failed += 1

    print(f"\n{'='*40}\n完成 - 成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    main()
