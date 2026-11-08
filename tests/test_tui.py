import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ensemble_atlas.cli import app
from ensemble_atlas.tui import SearchProcess, search_command, recording_lines, result_page, run


def hit(name):
    return {"original_name": name, "recording_id": name + "-id", "audio_path": "/audio/" + name,
            "score": 0.9, "matches": [{"start": 5, "end": 15, "score": 0.9},
                                      {"start": 10, "end": 20, "score": 0.8}]}


class TuiTests(unittest.TestCase):
    def test_search_dispatch_and_empty_index(self):
        with TemporaryDirectory() as directory:
            for mode, value in [("Text", "--a query with spaces"), ("Audio", "clip with spaces.wav")]:
                command = search_command(Path(directory), mode, value, "3", "2")
                self.assertEqual(command[:3], [sys.executable, "-m", "ensemble_atlas.cli"])
                self.assertIn(value, command)
                self.assertEqual(command[command.index("--top-k") + 1], "3")
            job = SearchProcess(search_command(Path(directory), "Text", "a dog barking", "5", "3"))
            try:
                self.assertEqual(job.process.wait(timeout=20), 0)
                self.assertEqual(json.loads(job.result()), [])
            finally:
                job.close()

    def test_validation(self):
        for value, count, windows in [("", "5", "3"), ("dog", "0", "3"),
                                       ("dog", "abc", "3"), ("dog", "3", "-1")]:
            with self.subTest(value=value, count=count, windows=windows), self.assertRaises(ValueError):
                search_command(Path("data"), "Text", value, count, windows)

    def test_cancel_reaps_child(self):
        job = SearchProcess([sys.executable, "-c", "import time; time.sleep(60)"])
        job.close()
        self.assertIsNotNone(job.process.poll())
        self.assertTrue(job.output.closed)

    def test_noninteractive_terminal(self):
        result = CliRunner().invoke(app, ["tui"])
        self.assertIsInstance(result.exception, ValueError)
        self.assertIn("interactive terminal", str(result.exception))

    def test_recording_pages_keep_timestamps_with_recording(self):
        hits = [hit("first.wav"), hit("second.wav"), hit("third.wav")]
        card = recording_lines(hits[0], 1, 72)
        height = len(card) + 3
        self.assertEqual([i for i, _ in result_page(hits, 0, 72, height)], [0])
        next_page = result_page(hits, 1, 72, height)
        self.assertEqual([i for i, _ in next_page], [1])
        self.assertIn("02  second.wav", next_page[0][1])
        self.assertTrue(any("00:00:05.000 - 00:00:15.000" in line for line in next_page[0][1]))
        self.assertEqual([i for i, _ in result_page(hits, 0, 72, height * 2)], [0, 1])
        # An oversized first card retains all its details for within-card pagination.
        narrow = result_page(hits, 0, 12, 3)
        self.assertEqual(len(narrow), 1)
        self.assertEqual(narrow[0][1], recording_lines(hits[0], 1, 12))

    def test_keyboard_moves_between_recordings(self):
        import curses
        screen = MagicMock()
        screen.getmaxyx.return_value = (24, 80)
        screen.get_wch.side_effect = ["a", "\n", curses.KEY_DOWN, curses.KEY_UP, "\x1b"]
        frames = []
        current = []
        screen.addnstr.side_effect = lambda row, column, text, *args: current.append(text)
        def refresh():
            frames.append("\n".join(current))
            current.clear()
        screen.refresh.side_effect = refresh
        job = MagicMock()
        job.process.poll.return_value = 0
        job.process.returncode = 0
        job.result.return_value = json.dumps([hit("first.wav"), hit("second.wav")])
        with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True), \
                patch("curses.wrapper", side_effect=lambda callback: callback(screen)), \
                patch("curses.has_colors", return_value=False), patch("curses.curs_set"), \
                patch("ensemble_atlas.tui.SearchProcess", return_value=job):
            run(Path("data"))
        self.assertIn("01  first.wav", frames[2])
        self.assertIn("02  second.wav", frames[3])
        self.assertNotIn("01  first.wav", frames[3])
        self.assertIn("01  first.wav", frames[4])
        job.close.assert_called_once()

    def test_json_cli_preserves_result_structure(self):
        with TemporaryDirectory() as directory, \
                patch("ensemble_atlas.cli.Storage") as storage, \
                patch("ensemble_atlas.cli.ClapEmbedder"), \
                patch("ensemble_atlas.cli.search_index", return_value=[hit("a.wav")]):
            storage.return_value.completed.return_value = [object()]
            result = CliRunner().invoke(app, ["--data-dir", directory, "search", "a sound", "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output), [hit("a.wav")])


if __name__ == "__main__":
    unittest.main()
