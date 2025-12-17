#!/usr/bin/env python3
"""vv-asr - 语音识别工具"""

# 在导入其他库前设置日志和警告过滤
import logging
import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*[Aa]ttention mask.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import argparse
import re
import shutil
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import torch

# DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_MODEL = "iic/SenseVoiceSmall"
DEFAULT_VAD_MAX_TIME = 30000
DEFAULT_BATCH_SIZE = 1

# 预编译正则表达式
_LANG_TAG_RE = re.compile(r"<\|(?:zh|en|ja|yue|ko)\|>")
_OTHER_TAG_RE = re.compile(r"<\|[^|]+\|>")


@lru_cache(maxsize=1)
def get_device() -> str:
    """自动检测最佳计算设备"""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """根据设备选择最佳数据类型"""
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def format_timestamp(ms: int) -> str:
    """毫秒转 HH:MM:SS.mmm"""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_qwen_model(model_name: str):
    """确保 Qwen3-0.6B 模型存在"""
    cache_dir = Path.home() / f".cache/modelscope/hub/models/{model_name}"
    qwen_dir = cache_dir / "Qwen3-0.6B"
    if qwen_dir.is_dir() and not (qwen_dir / "model.safetensors").exists():
        print("正在下载 Qwen3-0.6B 模型...")
        from huggingface_hub import snapshot_download
        hf_path = Path(snapshot_download(repo_id="Qwen/Qwen3-0.6B"))
        for f in hf_path.iterdir():
            if f.is_file() and not (qwen_dir / f.name).exists():
                shutil.copy2(f, qwen_dir / f.name)


def build_model(model_name: str, device: str, use_vad: bool, vad_max_time: int,
                use_compile: bool = True, use_half: bool = True):
    """构建ASR模型，分离 VAD 模型以获取时间戳"""
    from funasr import AutoModel

    # Fun-ASR-Nano 需要额外的 Qwen 模型
    if "Fun-ASR-Nano" in model_name:
        from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401 注册模型类
        ensure_qwen_model(model_name)

    dtype = get_dtype(device) if use_half else torch.float32
    dtype_name = str(dtype).split('.')[-1]
    print(f"加载模型: {model_name} (设备: {device}, 精度: {dtype_name})")

    # 分离加载 VAD 和 ASR 模型
    vad_model = None
    if use_vad:
        vad_model = AutoModel(model="fsmn-vad", device=device, disable_update=True)
        vad_model.vad_max_time = vad_max_time

    asr_model = AutoModel(model=model_name, device=device, disable_update=True)

    # 转换为半精度
    if use_half and dtype != torch.float32:
        try:
            asr_model.model = asr_model.model.to(dtype)
            print(f"已转换为 {dtype_name} 精度")
        except Exception as e:
            print(f"半精度转换失败: {e}")

    # torch.compile 优化 (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile") and device != "mps":
        try:
            asr_model.model = torch.compile(asr_model.model, mode="default")
            print("已启用 torch.compile 优化 (首次运行需要编译)")
        except Exception as e:
            print(f"torch.compile 失败: {e}")

    return {"asr": asr_model, "vad": vad_model}


def clean_text(text: str) -> list[str]:
    """清理 SenseVoice 输出，按识别的句子分行"""
    return [
        clean for part in _LANG_TAG_RE.split(text)
        if (clean := _OTHER_TAG_RE.sub("", part).strip())
    ]


def resample_audio(waveform, orig_sr: int, target_sr: int = 16000):
    """重采样音频到目标采样率"""
    if orig_sr == target_sr:
        return waveform
    import librosa
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def transcribe(models: dict, audio_path: str, **kwargs) -> dict:
    """识别单个音频，使用分离的 VAD 和 ASR 模型获取时间戳"""
    import soundfile as sf

    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    asr_model = models["asr"]
    vad_model = models.get("vad")

    segments = []
    all_text = []

    with torch.inference_mode():
        if vad_model:
            # 先用 VAD 模型获取语音片段时间戳
            vad_res = vad_model.generate(input=[audio_path], cache={})[0]
            vad_segments = vad_res.get("value", [])  # [[start_ms, end_ms], ...]

            if vad_segments:
                # 读取完整音频
                waveform, sample_rate = sf.read(audio_path)

                # 重采样到 16kHz (模型要求)
                target_sr = 16000
                if sample_rate != target_sr:
                    waveform = resample_audio(waveform, sample_rate, target_sr)
                    # 调整时间戳对应的采样点
                    sample_rate = target_sr

                # 对每个 VAD 片段进行 ASR 识别
                for seg in vad_segments:
                    start_ms, end_ms = seg[0], seg[1]
                    start_sample = int(start_ms * sample_rate / 1000)
                    end_sample = int(end_ms * sample_rate / 1000)

                    # 提取片段音频
                    segment_audio = waveform[start_sample:end_sample]
                    if len(segment_audio) == 0:
                        continue

                    # 识别片段
                    res = asr_model.generate(input=[segment_audio], cache={}, **kwargs)[0]
                    text = res.get("text", "")
                    cleaned = clean_text(text)

                    for sent in cleaned:
                        segments.append({"text": sent, "start": start_ms, "end": end_ms})
                        all_text.append(sent)
            else:
                # VAD 没有检测到语音，直接识别整个音频
                res = asr_model.generate(input=[audio_path], cache={}, **kwargs)[0]
                all_text = clean_text(res.get("text", ""))
        else:
            # 不使用 VAD，直接识别
            res = asr_model.generate(input=[audio_path], cache={}, **kwargs)[0]
            text = res.get("text", "")
            all_text = clean_text(text)

            # 尝试从结果获取时间戳
            if sentence_info := res.get("sentence_info"):
                segments = [
                    {"text": sent, "start": s["start"], "end": s["end"]}
                    for s in sentence_info
                    for sent in clean_text(s["text"])
                ]
            elif ts := res.get("timestamp"):
                segments = [{"text": sent, "start": ts[0][0], "end": ts[-1][1]} for sent in all_text]

    return {"text": "\n".join(all_text), "segments": segments}


def save_result(result: dict, path: Path):
    """保存结果到文件"""
    with path.open("w", encoding="utf-8") as f:
        if result["segments"]:
            lines = (
                f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text']}"
                for s in result["segments"]
            )
            f.write("\n".join(lines) + "\n")
        else:
            f.write(result["text"])


def main():
    parser = argparse.ArgumentParser(
        description="ASR工具 - 语音识别",
        epilog="示例: uv run python asr_tool.py audio.mp3 -o ./output")
    parser.add_argument("files", nargs="+", type=Path, help="音频文件")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("-l", "--language", default="auto", choices=["auto", "zh", "en", "ja"])
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hotwords", nargs="*")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--vad-max-time", type=int, default=DEFAULT_VAD_MAX_TIME)
    parser.add_argument("--no-itn", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile 优化")
    parser.add_argument("--no-half", action="store_true", help="禁用半精度 (float16/bfloat16)")
    parser.add_argument("--threads", type=int, help="CPU 线程数 (默认: 全部核心)")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else args.device

    num_threads = args.threads or os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"设备: {device}, CPU线程: {num_threads}")

    models = build_model(args.model, device, not args.no_vad, args.vad_max_time,
                         use_compile=not args.no_compile, use_half=not args.no_half)
    gen_kwargs = {"batch_size": args.batch_size, "language": args.language, "itn": not args.no_itn}
    if args.hotwords:
        gen_kwargs["hotwords"] = args.hotwords

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
