import os
import subprocess
import time


class AudioPlayer:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._current_song: str | None = None
        self._play_start_time: float = 0.0
        self._offset: float = 0.0  # accumulated playback position in seconds

    def play(self, song_path: str) -> bool:
        """Start or resume playing. Returns True if playback started."""
        if not os.path.exists(song_path):
            print(f"[audio] File not found: {song_path}")
            return False

        # Already playing this song
        if self._current_song == song_path and self._process and self._process.poll() is None:
            return False

        # Different song — reset offset
        if self._current_song != song_path:
            self._offset = 0.0

        self._kill_process()

        self._process = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
             "-ss", str(self._offset), song_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._play_start_time = time.time()
        self._current_song = song_path
        return True

    def is_playing(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self):
        """Pause playback and save the current position."""
        if self.is_playing():
            self._offset += time.time() - self._play_start_time
        self._kill_process()

    def _kill_process(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None
