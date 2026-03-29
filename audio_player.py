import os
import subprocess


class AudioPlayer:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._current_song: str | None = None

    def play(self, song_path: str) -> bool:
        """Start playing if not already playing this song."""
        if not os.path.exists(song_path):
            print(f"[audio] File not found: {song_path}")
            return False

        # Already playing this song
        if self._current_song == song_path and self._process and self._process.poll() is None:
            return False

        self._kill()

        self._process = subprocess.Popen(
            ["afplay", song_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current_song = song_path
        return True

    def is_playing(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self):
        self._kill()

    def _kill(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None
        self._current_song = None
