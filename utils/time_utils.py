import time


class Timer:
    def __init__(self):
        self._now = time.time()

    @staticmethod
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @property
    def elapsed(self):
        return self.format_time(time.time() - self._now)
