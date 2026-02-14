from __future__ import annotations

import argparse
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pyperclip
import sounddevice as sd
import ctranslate2
from faster_whisper import WhisperModel


class Logger:
    def __init__(self, use_color: bool = True, clear_after: int = 120) -> None:
        self.use_color = use_color and sys.stdout.isatty()
        self.clear_after = clear_after
        self._lines_since_clear = 0
        self._reset = "\033[0m" if self.use_color else ""
        self._colors = {
            "INFO": "\033[36m",    # cyan
            "WARN": "\033[33m",    # yellow
            "ERROR": "\033[31m",   # red
            "SUCCESS": "\033[32m", # green
            "SYSTEM": "\033[35m",  # magenta
        }

    def _clear_if_needed(self) -> None:
        if self.clear_after <= 0:
            return
        if self._lines_since_clear < self.clear_after:
            return
        if not sys.stdout.isatty():
            return
        # Clear terminal and reset cursor to top-left to keep long sessions readable.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        self._lines_since_clear = 0

    def _emit(self, level: str, message: str) -> None:
        color = self._colors.get(level, "")
        prefix = f"[{level}] "
        if self.use_color and color:
            print(f"{color}{prefix}{message}{self._reset}")
        else:
            print(f"{prefix}{message}")
        self._lines_since_clear += message.count("\n") + 1
        self._clear_if_needed()

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def success(self, message: str) -> None:
        self._emit("SUCCESS", message)

    def system(self, message: str) -> None:
        self._emit("SYSTEM", message)


class SoundNotifier:
    def __init__(self, samplerate: int = 24000, volume: float = 0.12) -> None:
        self.samplerate = samplerate
        self.volume = volume

    def _tone(self, freq_hz: float, duration_sec: float) -> np.ndarray:
        t = np.linspace(
            0.0,
            duration_sec,
            int(self.samplerate * duration_sec),
            endpoint=False,
            dtype=np.float32,
        )
        return (self.volume * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

    def _play(self, tone: np.ndarray, logger: Logger, blocking: bool = False) -> None:
        try:
            sd.stop()
            sd.play(tone, samplerate=self.samplerate, blocking=blocking)
        except Exception as exc:
            logger.warn(f"Sound playback unavailable: {exc}")

    def capture_started(self, logger: Logger) -> None:
        self._play(self._tone(freq_hz=920, duration_sec=0.08), logger, blocking=False)

    def capture_ended(self, logger: Logger) -> None:
        self._play(self._tone(freq_hz=620, duration_sec=0.08), logger, blocking=False)

    def transcription_complete(self, logger: Logger) -> None:
        self._play(self._tone(freq_hz=1180, duration_sec=0.12), logger, blocking=True)


def list_input_devices() -> list[tuple[int, dict]]:
    devices = sd.query_devices()
    results: list[tuple[int, dict]] = []
    for idx, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            results.append((idx, device))
    return results


def print_input_devices(logger: Logger) -> None:
    inputs = list_input_devices()
    if not inputs:
        logger.warn("No input microphones found.")
        return

    logger.system("Available microphones:")
    for idx, device in inputs:
        name = device.get("name", "Unknown")
        channels = int(device.get("max_input_channels", 0))
        rate = int(device.get("default_samplerate", 0))
        logger.info(f"[{idx}] {name} | channels={channels} | default_rate={rate}")


def choose_microphone_interactive(logger: Logger) -> int:
    inputs = list_input_devices()
    if not inputs:
        raise RuntimeError("No input microphones found.")

    print_input_devices(logger)
    valid_indices = {idx for idx, _ in inputs}

    while True:
        raw = input("Select microphone index: ").strip()
        if not raw:
            logger.warn("Please enter a microphone index.")
            continue
        if not raw.isdigit():
            logger.warn("Please enter a numeric index.")
            continue

        mic_index = int(raw)
        if mic_index not in valid_indices:
            logger.warn("Invalid microphone index. Pick one from the list.")
            continue

        return mic_index


def get_default_samplerate(mic_index: int) -> int:
    device = sd.query_devices(mic_index)
    rate = int(device.get("default_samplerate", 16000))
    return rate if rate > 0 else 16000


def _write_temp_wav(audio: np.ndarray, samplerate: int) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio[:, 0]

    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())

    return wav_path


def record_with_enter(
    mic_index: int,
    samplerate: int,
    logger: Logger,
    notifier: SoundNotifier,
) -> np.ndarray:
    frames: list[np.ndarray] = []

    def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
        if status:
            logger.warn(f"Audio warning: {status}")
        frames.append(indata.copy())

    input("Press Enter to start recording...")
    logger.info("Recording... press Enter again to stop.")
    notifier.capture_started(logger)

    with sd.InputStream(
        samplerate=samplerate,
        device=mic_index,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input()
    notifier.capture_ended(logger)

    if not frames:
        raise RuntimeError("No audio recorded.")

    return np.concatenate(frames, axis=0)


def _chunk_rms(chunk: np.ndarray) -> float:
    samples = np.asarray(chunk, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    if samples.ndim == 2:
        samples = samples[:, 0]
    return float(np.sqrt(np.mean(samples * samples)))


def record_ever_running(
    mic_index: int,
    samplerate: int,
    trigger_threshold: float,
    silence_threshold: float,
    silence_seconds: float,
    max_capture_seconds: float,
    chunk_ms: int,
    logger: Logger,
    notifier: SoundNotifier,
) -> np.ndarray:
    chunk_frames = max(1, int(samplerate * (chunk_ms / 1000.0)))
    max_chunks = max(1, int(max_capture_seconds / (chunk_ms / 1000.0)))
    silence_chunks_required = max(1, int(silence_seconds / (chunk_ms / 1000.0)))

    captured: list[np.ndarray] = []
    is_recording = False
    silence_chunks = 0

    logger.system(
        "Ever-running mode active. Listening continuously "
        f"(trigger_threshold={trigger_threshold:.4f})..."
    )

    with sd.InputStream(
        samplerate=samplerate,
        device=mic_index,
        channels=1,
        dtype="float32",
    ) as stream:
        while True:
            chunk, overflowed = stream.read(chunk_frames)
            if overflowed:
                logger.warn("Audio warning: input overflow detected.")

            level = _chunk_rms(chunk)

            if not is_recording:
                if level >= trigger_threshold:
                    is_recording = True
                    captured.append(chunk.copy())
                    logger.info(
                        f"Threshold exceeded (level={level:.4f}). "
                        "Capture started."
                    )
                    notifier.capture_started(logger)
                continue

            captured.append(chunk.copy())
            if len(captured) >= max_chunks:
                logger.warn("Reached max capture length. Stopping capture.")
                break

            if level < silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            if silence_chunks >= silence_chunks_required:
                logger.info("Detected sustained silence. Stopping capture.")
                break

    if not captured:
        raise RuntimeError("No audio captured in ever-running mode.")

    notifier.capture_ended(logger)
    return np.concatenate(captured, axis=0)


def transcribe_audio(
    model: WhisperModel,
    wav_path: Path,
    language: str | None,
    logger: Logger,
    notifier: SoundNotifier,
) -> str:
    logger.info("Starting transcription...")
    started_at = time.perf_counter()
    segments, _ = model.transcribe(
        str(wav_path),
        task="translate",
        language=language,
        vad_filter=True,
    )

    segment_list = list(segments)
    logger.success(
        f"Transcription finished. Generated {len(segment_list)} segments "
        f"(language={'auto' if language is None else language})."
    )
    transcription_seconds = time.perf_counter() - started_at
    logger.info(f"Transcription duration: {transcription_seconds:.2f}s")
    notifier.transcription_complete(logger)
    text = " ".join(segment.text.strip() for segment in segment_list).strip()
    return text


def resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "default":
        return compute_type

    supported: set[str] | None = None
    for target in (device, "auto"):
        try:
            supported = set(ctranslate2.get_supported_compute_types(target))
            break
        except Exception:
            continue

    if not supported:
        return "float32"

    if "float16" in supported:
        return "float16"
    if "float32" in supported:
        return "float32"
    return sorted(supported)[0]


def load_whisper_model(
    model_name: str,
    device: str,
    compute_type: str,
    logger: Logger,
) -> WhisperModel:
    logger.info(
        "Initializing Whisper model "
        f"(model={model_name}, device={device}, compute_type={compute_type})..."
    )
    started_at = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    load_seconds = time.perf_counter() - started_at
    logger.success("Whisper model loaded and ready.")
    logger.info(f"Model load duration: {load_seconds:.2f}s")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech-copy",
        description="Record from microphone and transcribe locally with faster-whisper.",
    )
    parser.add_argument("--list-mics", action="store_true", help="List microphones and exit.")
    parser.add_argument("--mic", type=int, help="Microphone device index.")
    parser.add_argument(
        "--mode",
        default="enter",
        choices=["enter", "ever-running"],
        help="Capture mode. 'enter' toggles with keyboard, 'ever-running' auto-triggers on mic level.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name for faster-whisper. Default: large-v3.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type. Example: float16, int8, int8_float16.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. en, tr). Default: auto-detect.",
    )
    parser.add_argument(
        "--auto-copy",
        dest="auto_copy",
        action="store_true",
        default=True,
        help="Automatically copy transcript to clipboard (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-copy",
        dest="auto_copy",
        action="store_false",
        help="Disable automatic transcript copy to clipboard.",
    )
    parser.add_argument(
        "--no-clipboard",
        dest="auto_copy",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.03,
        help="RMS level that starts capture in ever-running mode. Default: 0.03.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.015,
        help="RMS level considered silence during ever-running capture. Default: 0.015.",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=1.0,
        help="Stop capture after this much continuous silence in ever-running mode. Default: 1.0.",
    )
    parser.add_argument(
        "--max-capture-seconds",
        type=float,
        default=20.0,
        help="Maximum auto-capture length in ever-running mode. Default: 20.0.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Chunk size in ms for level checks in ever-running mode. Default: 100.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored logs.",
    )
    parser.add_argument(
        "--clear-after",
        type=int,
        default=120,
        help="Auto-clear terminal after this many log lines. Set 0 to disable. Default: 120.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = Logger(use_color=not args.no_color, clear_after=args.clear_after)
    notifier = SoundNotifier()

    if args.list_mics:
        print_input_devices(logger)
        return 0

    mic_index = args.mic if args.mic is not None else choose_microphone_interactive(logger)
    samplerate = get_default_samplerate(mic_index)
    compute_type = resolve_compute_type(args.device, args.compute_type)
    model = load_whisper_model(
        model_name=args.model,
        device=args.device,
        compute_type=compute_type,
        logger=logger,
    )
    logger.info(f"Using microphone index {mic_index} at {samplerate} Hz.")
    logger.info(f"Using compute type: {compute_type}")

    def handle_audio_capture(audio: np.ndarray) -> str:
        duration_seconds = len(audio) / samplerate if samplerate > 0 else 0
        logger.info(f"Captured {duration_seconds:.2f}s of audio. Preparing WAV...")

        wav_path = _write_temp_wav(audio=audio, samplerate=samplerate)

        try:
            logger.info(f"Transcribing with model '{args.model}' from {wav_path}...")
            return transcribe_audio(
                model=model,
                wav_path=wav_path,
                language=args.language,
                logger=logger,
                notifier=notifier,
            )
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

    if args.mode == "ever-running":
        logger.system("Ever-running loop started. Press Ctrl+C to stop.")
        try:
            while True:
                audio = record_ever_running(
                    mic_index=mic_index,
                    samplerate=samplerate,
                    trigger_threshold=args.trigger_threshold,
                    silence_threshold=args.silence_threshold,
                    silence_seconds=args.silence_seconds,
                    max_capture_seconds=args.max_capture_seconds,
                    chunk_ms=args.chunk_ms,
                    logger=logger,
                    notifier=notifier,
                )
                text = handle_audio_capture(audio)

                if not text:
                    logger.warn("No speech detected.")
                    continue

                logger.success("Transcript:")
                print(text)

                if args.auto_copy:
                    try:
                        pyperclip.copy(text)
                        logger.success("Copied transcript to clipboard.")
                    except pyperclip.PyperclipException as exc:
                        logger.warn(f"Could not copy to clipboard: {exc}")
        except KeyboardInterrupt:
            logger.system("Stopping ever-running mode.")
            return 0

    audio = record_with_enter(
        mic_index=mic_index,
        samplerate=samplerate,
        logger=logger,
        notifier=notifier,
    )
    text = handle_audio_capture(audio)

    if not text:
        logger.warn("No speech detected.")
        return 1

    logger.success("Transcript:")
    print(text)

    if args.auto_copy:
        try:
            pyperclip.copy(text)
            logger.success("Copied transcript to clipboard.")
        except pyperclip.PyperclipException as exc:
            logger.warn(f"Could not copy to clipboard: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
