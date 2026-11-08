"""Terminal search form backed by the existing atlas search command."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap


def search_command(data_dir: Path, mode: str, value: str, count: str, windows: str) -> list[str]:
    if not value.strip():
        raise ValueError("Enter search text or an audio file path.")
    try:
        limits = [int(count), int(windows)]
    except ValueError:
        raise ValueError("Result and window counts must be positive whole numbers.") from None
    if min(limits) < 1:
        raise ValueError("Result and window counts must be positive whole numbers.")
    command = [sys.executable, "-m", "ensemble_atlas.cli", "--data-dir", str(data_dir),
               "search", "--json", "--top-k", str(limits[0]), "--windows-per-recording", str(limits[1])]
    if mode == "Audio":
        command.extend(["--audio", str(Path(value).expanduser())])
    else:
        command.extend(["--", value])
    return command


class SearchProcess:
    """Use a file so verbose child output cannot block on a full pipe."""
    def __init__(self, command):
        self.output = tempfile.TemporaryFile()
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdout=self.output, stderr=self.errors,
                                            env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"})
        except BaseException:
            self.output.close()
            self.errors.close()
            raise

    def result(self):
        self.output.seek(0)
        return self.output.read().decode("utf-8", errors="replace")

    def error(self):
        self.errors.seek(0)
        return self.errors.read().decode("utf-8", errors="replace")

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.output.close()
            self.errors.close()


def recording_lines(hit: dict, rank: int, width: int) -> list[str]:
    """Wrap one whole recording, including every matching timestamp."""
    from .cli import timestamp
    content = [f"{rank:02d}  {hit['original_name']}",
               f"BEST COSINE  {hit['score']:.4f}",
               f"ID  {hit['recording_id']}", str(hit['audio_path'])]
    content += [f"{timestamp(match['start'])} - {timestamp(match['end'])}  |  {match['score']:.4f}"
                for match in hit['matches']]
    return [wrapped for line in content
            for wrapped in (textwrap.wrap(line, max(1, width)) or [""])]


def result_page(hits: list[dict], selected: int, width: int, height: int) -> list[tuple[int, list[str]]]:
    """Start at the selected recording; never split a following card across pages."""
    cards = []
    used = 0
    for index in range(selected, len(hits)):
        lines = recording_lines(hits[index], index + 1, width)
        size = len(lines) + 3
        if used + size > height and cards:
            break
        cards.append((index, lines))
        used += size
        if used >= height:
            break
    return cards


def run(data_dir: Path):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("atlas tui requires an interactive terminal.")
    try:
        import curses
    except ImportError:
        raise RuntimeError("This Python installation does not provide curses; use atlas search.") from None

    def screen_loop(screen):
        fields = ["Text", "", "5", "3"]
        focus = 1
        offset = 0
        message = "Enter a query, then press Enter to search."
        output = ""
        hits = []
        detail_page = 0
        job = None
        screen.timeout(100)
        screen.keypad(True)
        accent = muted = active = 0
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_CYAN, background)
            curses.init_pair(2, curses.COLOR_BLUE, background)
            curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_CYAN)
            accent, muted, active = (curses.color_pair(i) for i in (1, 2, 3))
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        try:
            while True:
                if job is not None and job.process.poll() is not None:
                    output = job.result()
                    message = ("Search complete." if job.process.returncode == 0 else
                               "Search failed. Edit the form to retry.")
                    if job.process.returncode == 0:
                        try:
                            hits = json.loads(output)
                            output = "" if hits else "No completed embeddings. Run atlas ingest and atlas worker first."
                        except (ValueError, TypeError):
                            hits = []
                            message = "Could not read search results."
                    else:
                        output = output or job.error()
                    job.close()
                    job = None
                    offset = 0
                    detail_page = 0
                height, width = screen.getmaxyx()
                screen.erase()

                def put(row, value, selected=False, style=0, column=2):
                    if 0 <= row < height - 1 and width > column + 1:
                        # Render control characters as spaces, never terminal commands.
                        safe = "".join(c if c.isprintable() else " " for c in value)
                        try:
                            screen.addnstr(row, column, safe, width - column - 1,
                                           (active or curses.A_REVERSE) if selected else style)
                        except curses.error:
                            pass

                put(0, " ENSEMBLE / ATLAS   •   Audio discovery ".ljust(max(1, width - 4)),
                    style=accent | curses.A_BOLD)
                put(1, f"Index  {data_dir.resolve()}", style=curses.A_DIM)
                labels = ["SOURCE  ←/→", "QUERY" if fields[0] == "Text" else "AUDIO PATH ≤10s",
                          "RECORDINGS", "TIMESTAMPS"]
                for index, label in enumerate(labels):
                    # Show the end of long fields while editing.
                    available = max(1, width - 24)
                    value = fields[index][-available:] or "Type here…"
                    if index == 0:
                        value = "[ TEXT ]   Audio" if fields[0] == "Text" else "Text   [ AUDIO ]"
                    put(index + 3, f" {label:<17} {value} ".ljust(max(1, width - 4)), focus == index,
                        style=curses.A_DIM if not fields[index] else 0)
                put(8, ("◌ Loading CLAP / searching…  Esc to cancel" if job else message), style=accent)
                counter = f"RECORDING {offset + 1} / {len(hits)}" if hits else "RESULTS"
                put(9, f"── {counter}  ·  cosine ranking, not confidence " + "─" * width, style=muted)
                page = max(1, height - 13)
                offset = min(offset, max(0, len(hits) - 1))
                cards = result_page(hits, offset, max(1, width - 8), page)
                row = 10
                for index, lines in cards:
                    oversized = len(lines) + 3 > page
                    if oversized:
                        capacity = max(1, page - 3)
                        pages = max(1, (len(lines) + capacity - 1) // capacity)
                        detail_page = min(detail_page, pages - 1)
                        lines = lines[detail_page * capacity:(detail_page + 1) * capacity]
                        title = f" DETAILS {detail_page + 1}/{pages} · F3/F4 "
                    else:
                        title = " SELECTED " if index == offset else ""
                    put(row, "┌" + title + "─" * max(0, width - 6 - len(title)) + "┐", style=accent)
                    row += 1
                    for number, line in enumerate(lines):
                        if row >= height - 4:
                            break
                        put(row, "│", style=accent)
                        put(row, line, style=curses.A_BOLD if number == 0 else 0, column=4)
                        put(row, "│", style=accent, column=width - 3)
                        row += 1
                    put(row, "└" + "─" * max(0, width - 6) + "┘", style=accent)
                    row += 2
                if not hits:
                    text = output or "Find a sound. Discover where it happens.\nSearch by description or a short audio example."
                    lines = [part for line in text.splitlines()
                             for part in (textwrap.wrap(line, max(1, width - 8)) or [""])]
                    for row, line in enumerate(lines[:page], 10):
                        put(row, line, style=curses.A_DIM)
                put(height - 3, "↑/↓ Recording  PgUp/PgDn Page  Tab Field  Enter Search", style=accent)
                put(height - 2, "Esc Cancel/Exit  Ctrl-U Clear  Ctrl-C Quit", style=curses.A_DIM)
                if height < 20 or width < 48:
                    screen.erase()
                    put(0, "Enlarge terminal to at least 48 columns × 20 rows.", column=0)
                    put(1, "Esc / Ctrl-C to exit.", column=0)
                screen.refresh()
                try:
                    key = screen.get_wch()
                except curses.error:
                    continue
                if key == "\x03":
                    break
                if key == "\x1b":
                    if job:
                        job.close()
                        job = None
                        message = "Search cancelled."
                    else:
                        break
                elif key == "\t":
                    focus = (focus + 1) % 4
                elif key == curses.KEY_BTAB:
                    focus = (focus - 1) % 4
                elif key in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
                    step = max(1, len(cards))
                    offset = min(max(0, len(hits) - 1), max(0, offset + {
                        curses.KEY_UP: -1, curses.KEY_DOWN: 1,
                        curses.KEY_PPAGE: -step, curses.KEY_NPAGE: step}[key]))
                    detail_page = 0
                elif key in (curses.KEY_F3, curses.KEY_F4):
                    detail_page = max(0, detail_page + (1 if key == curses.KEY_F4 else -1))
                elif focus == 0 and key in (curses.KEY_LEFT, curses.KEY_RIGHT, " "):
                    fields[0] = "Audio" if fields[0] == "Text" else "Text"
                    fields[1] = ""
                elif key in ("\n", "\r", curses.KEY_ENTER) and job is None:
                    try:
                        job = SearchProcess(search_command(data_dir, *fields))
                        output = ""
                        hits = []
                        offset = 0
                        detail_page = 0
                    except (ValueError, OSError) as error:
                        message = str(error)
                elif focus > 0:
                    if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                        fields[focus] = fields[focus][:-1]
                    elif key == "\x15":
                        fields[focus] = ""
                    elif isinstance(key, str) and key.isprintable():
                        fields[focus] += key
        finally:
            if job:
                job.close()

    try:
        curses.wrapper(screen_loop)
    except curses.error as error:
        raise RuntimeError(f"Cannot initialize terminal interface: {error}") from None
