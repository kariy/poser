import time
import os
import pygame


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self._last_triggered: dict[str, float] = {}

    def play_if_ready(self, pose_name: str, song_path: str, cooldown: float) -> bool:
        """Play the song if cooldown has elapsed. Returns True if playback started."""
        if not os.path.exists(song_path):
            return False

        now = time.time()
        last = self._last_triggered.get(pose_name, 0)

        if now - last < cooldown:
            return False

        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        self._last_triggered[pose_name] = now
        return True

    def is_playing(self) -> bool:
        return pygame.mixer.music.get_busy()

    def stop(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()
