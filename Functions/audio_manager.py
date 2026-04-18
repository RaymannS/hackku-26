import os
import pygame

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio"))
TRACK_FILES = {
    "normal": os.path.join(AUDIO_DIR, "normal.mp3"),
    "orc": os.path.join(AUDIO_DIR, "orc.mp3"),
    "boss": os.path.join(AUDIO_DIR, "boss.mp3"),
}
DEFAULT_VOLUME = 0.5


class AudioManager:
    def __init__(self):
        self.enabled = False
        self.current_state = None
        self._init_mixer()

    def _init_mixer(self):
        try:
            pygame.mixer.init()
            self.enabled = True
        except Exception as exc:
            print(f"AudioManager: could not initialize audio: {exc}")

    def _track_path(self, state):
        return TRACK_FILES.get(state)

    def _load_track(self, path):
        if not os.path.exists(path):
            print(f"AudioManager: missing audio file: {path}")
            return False
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(DEFAULT_VOLUME)
            return True
        except Exception as exc:
            print(f"AudioManager: failed to load track {path}: {exc}")
            return False

    def play(self, state):
        if not self.enabled:
            return
        if self.current_state == state:
            return
        path = self._track_path(state)
        if not path:
            return
        if not self._load_track(path):
            return
        try:
            pygame.mixer.music.play(-1)
            self.current_state = state
            print(f"AudioManager: playing '{state}' music")
        except Exception as exc:
            print(f"AudioManager: failed to play music for state '{state}': {exc}")

    def play_normal(self):
        self.play("normal")

    def play_orc(self):
        self.play("orc")

    def play_boss(self):
        self.play("boss")

    def stop(self):
        if self.enabled:
            pygame.mixer.music.stop()
            self.current_state = None


audio_manager = AudioManager()
