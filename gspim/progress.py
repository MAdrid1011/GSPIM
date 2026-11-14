"""Small terminal progress display for long reproduction stages."""
import sys
import threading
import time


def bar(done, total, width=20):
    filled = min(width, max(0, width * done // total))
    return '[' + '#' * filled + '-' * (width - filled) + f'] {done}/{total}'


class Progress:
    def __init__(self, label, total=1, completed=None, *, interval=None, stream=None):
        self.label, self.total, self.completed = label, total, completed
        self.stream = sys.stderr if stream is None else stream
        self.tty = self.stream.isatty()
        self.interval = interval if interval is not None else (2 if self.tty else 20)
        self.stop = threading.Event()
        self.current = 0
        self.last_display = None

    def set(self, label, done, total):
        self.label, self.current, self.total = label, done, total

    def display(self, finished=False, failed=False):
        done = (self.total if finished else self.completed() if self.completed else self.current)
        elapsed = int(time.monotonic() - self.started)
        text = f'{self.label} {bar(done, self.total)}  {elapsed//60:02}:{elapsed%60:02}'
        if failed:
            text += ' failed'
        if not self.tty and text == self.last_display:
            return
        self.last_display = text
        self.stream.write(('\r\033[2K' if self.tty else '') + text + ('\n' if finished or failed or not self.tty else ''))
        self.stream.flush()

    def update(self):
        while not self.stop.wait(self.interval):
            self.display()

    def __enter__(self):
        self.started = time.monotonic()
        self.display()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        self.display(finished=kind is None, failed=kind is not None)
