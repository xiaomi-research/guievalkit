import time


class Timer:
    def __init__(self):
        self._start_time = time.perf_counter()

    def reset(self):
        """Reset the timer's start time."""
        self._start_time = time.perf_counter()

    @staticmethod
    def format_time(seconds):
        """Format total seconds into HH:MM:SS string."""
        hours, rem = divmod(int(seconds), 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @property
    def elapsed_seconds(self) -> float:
        """Return raw elapsed seconds as float."""
        return time.perf_counter() - self._start_time

    @property
    def elapsed(self) -> str:
        """Return formatted elapsed time string (HH:MM:SS)."""
        return self.format_time(self.elapsed_seconds)
