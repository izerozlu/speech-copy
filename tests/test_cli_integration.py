import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whisper_cli import cli


class TestCliIntegration(unittest.TestCase):
    @patch("whisper_cli.cli.print_input_devices")
    def test_list_mics_mode_exits_without_loading_model(self, mock_print_input_devices: MagicMock) -> None:
        with patch.object(sys, "argv", ["main.py", "--list-mics"]):
            result = cli.main()

        self.assertEqual(result, 0)
        mock_print_input_devices.assert_called_once()

    @patch("whisper_cli.cli.pyperclip.copy")
    @patch("whisper_cli.cli.transcribe_audio")
    @patch("whisper_cli.cli._write_temp_wav")
    @patch("whisper_cli.cli.record_with_enter")
    @patch("whisper_cli.cli.load_whisper_model")
    @patch("whisper_cli.cli.resolve_compute_type", return_value="float32")
    @patch("whisper_cli.cli.get_default_samplerate", return_value=16000)
    def test_enter_mode_single_capture_flow(
        self,
        _mock_samplerate: MagicMock,
        _mock_compute_type: MagicMock,
        mock_load_model: MagicMock,
        mock_record_with_enter: MagicMock,
        mock_write_temp_wav: MagicMock,
        mock_transcribe_audio: MagicMock,
        mock_clipboard_copy: MagicMock,
    ) -> None:
        fake_audio = np.zeros((1600, 1), dtype=np.float32)
        fake_model = object()
        fake_wav = Path("/tmp/fake-enter.wav")

        mock_load_model.return_value = fake_model
        mock_record_with_enter.return_value = fake_audio
        mock_write_temp_wav.return_value = fake_wav
        mock_transcribe_audio.return_value = "hello from enter mode"

        with patch.object(sys, "argv", ["main.py", "--mic", "1"]):
            result = cli.main()

        self.assertEqual(result, 0)
        mock_load_model.assert_called_once()
        mock_record_with_enter.assert_called_once()
        mock_transcribe_audio.assert_called_once()
        mock_clipboard_copy.assert_called_once_with("hello from enter mode")

    @patch("whisper_cli.cli.pyperclip.copy")
    @patch("whisper_cli.cli.transcribe_audio")
    @patch("whisper_cli.cli._write_temp_wav")
    @patch("whisper_cli.cli.record_with_enter")
    @patch("whisper_cli.cli.load_whisper_model")
    @patch("whisper_cli.cli.resolve_compute_type", return_value="float32")
    @patch("whisper_cli.cli.get_default_samplerate", return_value=16000)
    def test_enter_mode_no_auto_copy_disables_clipboard(
        self,
        _mock_samplerate: MagicMock,
        _mock_compute_type: MagicMock,
        mock_load_model: MagicMock,
        mock_record_with_enter: MagicMock,
        mock_write_temp_wav: MagicMock,
        mock_transcribe_audio: MagicMock,
        mock_clipboard_copy: MagicMock,
    ) -> None:
        fake_audio = np.zeros((1600, 1), dtype=np.float32)
        fake_model = object()
        fake_wav = Path("/tmp/fake-enter.wav")

        mock_load_model.return_value = fake_model
        mock_record_with_enter.return_value = fake_audio
        mock_write_temp_wav.return_value = fake_wav
        mock_transcribe_audio.return_value = "hello from enter mode"

        with patch.object(sys, "argv", ["main.py", "--mic", "1", "--no-auto-copy"]):
            result = cli.main()

        self.assertEqual(result, 0)
        mock_load_model.assert_called_once()
        mock_record_with_enter.assert_called_once()
        mock_transcribe_audio.assert_called_once()
        mock_clipboard_copy.assert_not_called()

    @patch("whisper_cli.cli.pyperclip.copy")
    @patch("whisper_cli.cli.transcribe_audio")
    @patch("whisper_cli.cli._write_temp_wav")
    @patch("whisper_cli.cli.record_ever_running")
    @patch("whisper_cli.cli.load_whisper_model")
    @patch("whisper_cli.cli.resolve_compute_type", return_value="float32")
    @patch("whisper_cli.cli.get_default_samplerate", return_value=16000)
    def test_ever_running_mode_loops_until_ctrl_c(
        self,
        _mock_samplerate: MagicMock,
        _mock_compute_type: MagicMock,
        mock_load_model: MagicMock,
        mock_record_ever_running: MagicMock,
        mock_write_temp_wav: MagicMock,
        mock_transcribe_audio: MagicMock,
        mock_clipboard_copy: MagicMock,
    ) -> None:
        fake_audio = np.zeros((800, 1), dtype=np.float32)
        fake_model = object()
        fake_wav = Path("/tmp/fake-ever.wav")

        mock_load_model.return_value = fake_model
        mock_record_ever_running.side_effect = [fake_audio, KeyboardInterrupt()]
        mock_write_temp_wav.return_value = fake_wav
        mock_transcribe_audio.return_value = "hello from ever running"

        with patch.object(sys, "argv", ["main.py", "--mode", "ever-running", "--mic", "1"]):
            result = cli.main()

        self.assertEqual(result, 0)
        self.assertEqual(mock_record_ever_running.call_count, 2)
        mock_transcribe_audio.assert_called_once()
        mock_clipboard_copy.assert_called_once_with("hello from ever running")


if __name__ == "__main__":
    unittest.main()
