import numpy as np
import cv2
import random
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Import render functions
try:
    from .render import draw_orc, draw_label
except ImportError:
    # For testing purposes
    def draw_orc(*args): pass
    def draw_label(*args): pass

class CharacterType(Enum):
    ORC = "orc"
    ELF = "elf"
    HUMAN = "human"
    DWARF = "dwarf"

@dataclass
class CharacterAppearance:
    body_color: Tuple[int, int, int]
    head_color: Tuple[int, int, int]
    eye_color: Tuple[int, int, int]
    size: int
    special_features: Optional[Callable] = None  # For custom drawing like tusks

@dataclass
class CharacterStats:
    health: int
    strength: int
    agility: int
    intelligence: int

@dataclass
class CharacterConfig:
    name: str
    appearance: CharacterAppearance
    stats: CharacterStats
    spawn_chance: float  # Probability weight for spawning
    draw_function: Callable  # Function to draw the character

class CharacterGenerator:
    def __init__(self):
        self.character_configs: Dict[CharacterType, CharacterConfig] = {}
        self.active_characters: List[Dict] = []  # List of spawned characters
        self._setup_default_characters()

    def _setup_default_characters(self):
        # Default orc configuration
        orc_appearance = CharacterAppearance(
            body_color=(30, 90, 30),
            head_color=(40, 110, 40),
            eye_color=(0, 0, 180),
            size=12,
            special_features=self._draw_orc_features
        )
        orc_stats = CharacterStats(
            health=100,
            strength=15,
            agility=8,
            intelligence=6
        )
        orc_config = CharacterConfig(
            name="Orc",
            appearance=orc_appearance,
            stats=orc_stats,
            spawn_chance=1.0,
            draw_function=self._draw_orc
        )
        self.add_character_type(CharacterType.ORC, orc_config)

    def add_character_type(self, char_type: CharacterType, config: CharacterConfig):
        """Add a new character type for future spawning."""
        self.character_configs[char_type] = config

    def _draw_orc(self, canvas, x, y, config: CharacterConfig):
        """Draw an orc character."""
        size = config.appearance.size
        # Body
        cv2.circle(canvas, (x, y), size // 2, config.appearance.body_color, -1)
        # Head
        cv2.circle(canvas, (x, y - size), size // 3, config.appearance.head_color, -1)
        # Eyes
        cv2.circle(canvas, (x - 3, y - size), 2, config.appearance.eye_color, -1)
        cv2.circle(canvas, (x + 3, y - size), 2, config.appearance.eye_color, -1)
        # Special features (tusks)
        if config.appearance.special_features:
            config.appearance.special_features(canvas, x, y, size)

    def _draw_orc_features(self, canvas, x, y, size):
        """Draw orc-specific features like tusks."""
        cv2.line(canvas, (x - 3, y - size + 3), (x - 5, y - size + 7), (200, 200, 200), 1)
        cv2.line(canvas, (x + 3, y - size + 3), (x + 5, y - size + 7), (200, 200, 200), 1)

    def spawn_characters(self, canvas, player_x: int, player_y: int, char_type: CharacterType, n: int = 5, radius: int = 60) -> List[Dict]:
        """Spawn characters of a specific type near the player."""
        if char_type not in self.character_configs:
            print(f"Character type {char_type} not configured.")
            return []

        config = self.character_configs[char_type]
        spawned = 0
        attempts = 0
        placed = []
        while spawned < n and attempts < 200:
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(radius * 0.3, radius)
            x = int(player_x + dist * np.cos(angle))
            y = int(player_y + dist * np.sin(angle))
            h, w = canvas.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                # Check distance from existing characters
                if all(abs(x - char["x"]) > 15 or abs(y - char["y"]) > 15 for char in placed):
                    config.draw_function(canvas, x, y, config)
                    char_data = {
                        "x": x,
                        "y": y,
                        "type": char_type,
                        "defeated": False,
                        "stats": config.stats,
                        "name": config.name
                    }
                    placed.append(char_data)
                    spawned += 1
            attempts += 1

        print(f"Spawned {spawned} {config.name}s near ({player_x}, {player_y})")
        self.active_characters.extend(placed)
        return placed

    def clear_characters(self, char_type: Optional[CharacterType] = None):
        """Clear all characters or characters of a specific type."""
        if char_type:
            self.active_characters = [char for char in self.active_characters if char["type"] != char_type]
        else:
            self.active_characters.clear()

    def get_characters_near(self, x: int, y: int, radius: int = 50) -> List[Dict]:
        """Get all active characters within a radius of a point."""
        return [char for char in self.active_characters 
                if np.sqrt((char["x"] - x)**2 + (char["y"] - y)**2) <= radius]

    def defeat_character(self, index: int):
        """Mark a character as defeated."""
        if 0 <= index < len(self.active_characters):
            self.active_characters[index]["defeated"] = True

    def draw_all_characters(self, canvas):
        """Redraw all active characters on the canvas."""
        for char in self.active_characters:
            if not char["defeated"]:
                config = self.character_configs[char["type"]]
                config.draw_function(canvas, char["x"], char["y"], config)