import numpy as np
import cv2
import random
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Import render functions
try:
    from .render import draw_tree, draw_cactus, draw_house, draw_label, apply_desert_terrain
except ImportError:
    # For testing purposes
    def draw_tree(*args): pass
    def draw_cactus(*args): pass  
    def draw_house(*args): pass
    def draw_label(*args): pass
    def apply_desert_terrain(*args): pass

class TerrainType(Enum):
    WATER = "water"
    LAND = "land"
    MOUNTAIN = "mountain"

class ItemType(Enum):
    FOREST = "forest"
    DESERT = "desert"
    TOWN = "town"
    VILLAGE = "village"
    CITY = "city"
    ORC = "orc"

@dataclass
class BandConfig:
    in_min: int
    in_max: int
    out_min: int
    out_max: int
    colormap: int

@dataclass
class TerrainConfig:
    height_levels: Dict[str, int]
    band_configs: Dict[TerrainType, BandConfig]

    @classmethod
    def default(cls) -> 'TerrainConfig':
        return cls(
            height_levels={
                "sea_level": 90,
                "mountain_level": 180,
                "snow_level": 240
            },
            band_configs={
                TerrainType.WATER: BandConfig(0, 90, 0, 170, cv2.COLORMAP_PARULA),
                TerrainType.LAND: BandConfig(90, 180, 60, 120, cv2.COLORMAP_DEEPGREEN),
                TerrainType.MOUNTAIN: BandConfig(180, 255, 25, 85, cv2.COLORMAP_BONE)
            }
        )

@dataclass
class ItemConfig:
    terrain_mask: Callable[[np.ndarray, int, int, int], np.ndarray]
    count_range: Tuple[int, int]
    min_distance: int
    draw_function: Callable
    size_range: Optional[Tuple[int, int]] = None

class SceneGenerator:
    def __init__(self, terrain_config: Optional[TerrainConfig] = None):
        self.terrain_config = terrain_config or TerrainConfig.default()
        self.item_configs = self._setup_item_configs()

    def _setup_item_configs(self) -> Dict[ItemType, ItemConfig]:
        """Setup default item configurations."""
        return {
            ItemType.FOREST: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea) & (Z < mountain),
                count_range=(25, 40),
                min_distance=20,
                draw_function=lambda canvas, x, y: draw_tree(canvas, x, y)
            ),
            ItemType.DESERT: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea) & (Z < sea + 50),
                count_range=(30, 45),
                min_distance=25,
                draw_function=lambda canvas, x, y: draw_cactus(canvas, x, y)
            ),
            ItemType.TOWN: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < mountain - 20),
                count_range=(3, 8),
                min_distance=15,
                draw_function=lambda canvas, x, y: draw_house(canvas, x, y)
            ),
            ItemType.VILLAGE: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < mountain - 20),
                count_range=(2, 5),
                min_distance=12,
                draw_function=lambda canvas, x, y: draw_house(canvas, x, y, size=10)
            ),
            ItemType.CITY: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < mountain - 20),
                count_range=(8, 15),
                min_distance=18,
                draw_function=lambda canvas, x, y: draw_house(canvas, x, y, size=16)
            )
        }

    def generate_terrain(self, depth_path: str) -> Tuple[np.ndarray, ...]:
        """Generate base terrain layers from depth data."""
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth file: {depth_path}")

        Z = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate slope (cliffs)
        gy, gx = np.gradient(Z.astype(float))
        slope = np.sqrt(gx**2 + gy**2)
        slope = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply band colorization
        bands = {}
        for terrain_type, config in self.terrain_config.band_configs.items():
            bands[terrain_type] = self._apply_band(Z, config)

        # Create masks
        levels = self.terrain_config.height_levels
        water_mask = Z < levels["sea_level"]
        mountain_mask = Z >= levels["mountain_level"]
        snow_mask = Z >= levels["snow_level"]
        cliffs = slope > 120

        # Base composition
        final = bands[TerrainType.LAND].copy()
        final[water_mask] = bands[TerrainType.WATER][water_mask]
        final[mountain_mask] = bands[TerrainType.MOUNTAIN][mountain_mask]
        final[snow_mask] = [245, 245, 245]  # Snow color
        final[cliffs & ~water_mask] = (final[cliffs & ~water_mask] * 0.6).astype(np.uint8)

        # Parchment overlay
        paper = np.full_like(final, [235, 220, 190])
        final = cv2.addWeighted(final, 0.85, paper, 0.15, 0)

        # Edge outlines
        edges = cv2.Canny(Z, 40, 120)
        final[edges > 0] = [20, 20, 20]

        return (Z, bands[TerrainType.WATER], bands[TerrainType.LAND],
                bands[TerrainType.MOUNTAIN], water_mask, mountain_mask,
                snow_mask, cliffs, final)

    def _apply_band(self, depth: np.ndarray, config: BandConfig) -> np.ndarray:
        """Apply color band transformation."""
        x = np.clip(depth - config.in_min, 0, config.in_max - config.in_min)
        x = x / (config.in_max - config.in_min + 1e-6)
        x = x * (config.out_max - config.out_min) + config.out_min
        x = x.astype(np.uint8)
        return cv2.applyColorMap(x, config.colormap)

    def generate_items(self, item_type: ItemType, canvas: np.ndarray, Z: np.ndarray,
                      region_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Generate items of specified type on canvas."""
        if item_type not in self.item_configs:
            raise ValueError(f"Unknown item type: {item_type}")

        config = self.item_configs[item_type]
        levels = self.terrain_config.height_levels

        # Create terrain suitability mask
        terrain_mask = config.terrain_mask(Z, levels["sea_level"],
                                         levels["mountain_level"],
                                         levels["snow_level"])

        # Apply region mask if provided
        if region_mask is not None:
            terrain_mask = terrain_mask & region_mask

        # Generate item positions
        count = random.randint(*config.count_range)
        positions = self._scatter_placer(terrain_mask, count, config.min_distance)

        # Draw items
        for x, y in positions:
            config.draw_function(canvas, x, y)

        return positions

    def _scatter_placer(self, mask: np.ndarray, n: int, min_dist: int) -> List[Tuple[int, int]]:
        """Scatter n points on mask with minimum distance constraints."""
        coords = np.argwhere(mask).tolist()
        if not coords:
            return []

        random.shuffle(coords)
        placed = []

        for y, x in coords:
            if len(placed) >= n:
                break
            if all(abs(x - px) >= min_dist and abs(y - py) >= min_dist for px, py in placed):
                placed.append((x, y))

        return placed

    def add_item_type(self, item_type: ItemType, config: ItemConfig):
        """Add or override an item type configuration.
        
        Example usage:
        >>> from Functions.scene_generator import SceneGenerator, ItemType, ItemConfig
        >>> scene_gen = SceneGenerator()
        >>> dungeon_config = ItemConfig(
        ...     terrain_mask=lambda Z, sea, mountain, snow: (Z >= mountain - 20) & (Z < mountain),
        ...     count_range=(1, 3),
        ...     min_distance=50,
        ...     draw_function=lambda canvas, x, y: draw_dungeon(canvas, x, y)
        ... )
        >>> scene_gen.add_item_type(ItemType.DUNGEON, dungeon_config)
        """
        self.item_configs[item_type] = config

    def update_terrain_config(self, height_levels: Optional[Dict[str, int]] = None, 
                             band_configs: Optional[Dict[TerrainType, BandConfig]] = None):
        """Update terrain configuration parameters.
        
        Example usage:
        >>> scene_gen.update_terrain_config(
        ...     height_levels={"sea_level": 100, "mountain_level": 200},
        ...     band_configs={
        ...         TerrainType.WATER: BandConfig(0, 100, 0, 180, cv2.COLORMAP_OCEAN)
        ...     }
        ... )
        """
        if height_levels:
            self.terrain_config.height_levels.update(height_levels)
        if band_configs:
            self.terrain_config.band_configs.update(band_configs)

    def get_available_item_types(self) -> List[ItemType]:
        """Get list of available item types."""
        return list(self.item_configs.keys())