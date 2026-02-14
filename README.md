# Speech Copy

A local speech-to-text CLI using `faster-whisper` and your microphone.

## Features

- Interactive microphone selection (or pass `--mic`)
- Enter-to-toggle recording (press Enter to start, Enter again to stop)
- Ever-running mode: continuously monitors mic and auto-starts capture on sound threshold
- Short ping sounds for capture start, capture end, and transcription complete
- Local transcription with Whisper `large-v3` by default
- Transcript output copied to clipboard automatically

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional flags:

```bash
python main.py --list-mics
python main.py --mic 2
python main.py --model large-v3
python main.py --device cpu --compute-type int8
python main.py --auto-copy
python main.py --no-auto-copy
python main.py --mode ever-running --trigger-threshold 0.03
python main.py --mode ever-running --trigger-threshold 0.02 --silence-seconds 1.2
python main.py --clear-after 80
python main.py --no-color --clear-after 0
```

## Notes

- First run downloads the selected model, so startup can take longer.
- Clipboard auto-copy is enabled by default.

## Clipboard Behavior

- `--auto-copy`: Explicitly enable automatic transcript copy (default behavior).
- `--no-auto-copy`: Disable automatic transcript copy.
- `--no-clipboard`: Legacy alias for `--no-auto-copy` (kept for compatibility).

## Ever-Running Mode Configuration

Use `--mode ever-running` to keep the microphone listener active continuously.  
The app will wait for sound above a threshold, capture speech, transcribe it, then go back to waiting.

Core tuning flags:

- `--trigger-threshold` (default: `0.03`): RMS level that starts a capture.
  - Increase it if background noise triggers false captures.
  - Decrease it if quiet speech is not detected.
- `--silence-threshold` (default: `0.015`): RMS level treated as silence after capture starts.
- `--silence-seconds` (default: `1.0`): How long silence must continue before capture stops.
- `--max-capture-seconds` (default: `20.0`): Hard cap for one capture to avoid very long recordings.
- `--chunk-ms` (default: `100`): Analysis window size for level checks (lower = more reactive).

Recommended starting point:

```bash
python main.py --mode ever-running \
  --trigger-threshold 0.03 \
  --silence-threshold 0.015 \
  --silence-seconds 1.0 \
  --max-capture-seconds 20 \
  --chunk-ms 100
```

Noisy room example:

```bash
python main.py --mode ever-running \
  --trigger-threshold 0.05 \
  --silence-threshold 0.025 \
  --silence-seconds 0.8
```

Quiet speaker example:

```bash
python main.py --mode ever-running \
  --trigger-threshold 0.02 \
  --silence-threshold 0.01 \
  --silence-seconds 1.2
```

Long sessions:

- `--clear-after` default is `120` lines; set lower (for example `80`) to clear more often.
- Disable clearing with `--clear-after 0`.
- `--no-color` default is off (colors enabled on TTY output); pass it to disable colors.
