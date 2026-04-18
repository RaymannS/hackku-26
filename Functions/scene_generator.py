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
    DEEP_WATER = "deep_water"
    SHALLOW_WATER = "shallow_water"
    COASTAL = "coastal"
    LOWLANDS = "lowlands"
    HILLS = "hills"
    HIGHLANDS = "highlands"
    MOUNTAINS = "mountains"
    PEAKS = "peaks"

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
    start_color: Tuple[int, int, int]  # RGB start color
    end_color: Tuple[int, int, int]    # RGB end color

@dataclass
class TerrainConfig:
    height_levels: Dict[str, int]
    band_configs: Dict[TerrainType, BandConfig]

    # The order of terrain bands is fixed and is used to generate the final map.
    BAND_ORDER = [
        TerrainType.DEEP_WATER,
        TerrainType.SHALLOW_WATER,
        TerrainType.COASTAL,
        TerrainType.LOWLANDS,
        TerrainType.HILLS,
        TerrainType.HIGHLANDS,
        TerrainType.MOUNTAINS,
        TerrainType.PEAKS
    ]

    # Threshold key names must match these band endings.
    BAND_THRESHOLD_KEYS = [
        "deep_water_level",
        "shallow_water_level",
        "coastal_level",
        "lowlands_level",
        "hills_level",
        "highlands_level",
        "mountains_level",
        "peaks_level"
    ]

    # Colors are defined here; modify these tuples to change the gradient for each band.
    BAND_COLOR_MAP = {
        TerrainType.DEEP_WATER: ((10, 20, 60), (15, 30, 80)),          # Very deep blue
        TerrainType.SHALLOW_WATER: ((15, 30, 80), (30, 60, 120)),     # Deep blue to medium blue
        TerrainType.COASTAL: ((210, 180, 140), (139, 90, 43)),         # Sandy yellow to brown
        TerrainType.LOWLANDS: ((85, 107, 47), (107, 142, 35)),        # Olive green to yellow-green
        TerrainType.HILLS: ((107, 142, 35), (34, 139, 34)),           # Yellow-green to forest green
        TerrainType.HIGHLANDS: ((34, 139, 34), (25, 100, 25)),        # Forest green to dark green
        TerrainType.MOUNTAINS: ((80, 80, 80), (120, 120, 120)),       # Gray to light gray
        TerrainType.PEAKS: ((120, 120, 120), (240, 240, 240))         # Light gray to snow white
    }

    @classmethod
    def default(cls) -> 'TerrainConfig':
        # Change elevation thresholds here. Only this block is required to adjust band heights.
        height_levels = {
            "deep_water_level": 45,      # Max elevation of deep ocean
            "shallow_water_level": 78,   # Max elevation of shallow ocean
            "coastal_level": 88,         # Max elevation of beaches/coastal plains
            "lowlands_level": 115,       # Max elevation of lowland plains
            "hills_level": 150,          # Max elevation of rolling hills
            "highlands_level": 175,      # Max elevation of highlands
            "mountains_level": 220,      # Max elevation of mountain slopes
            "peaks_level": 255           # Top elevation for snowy peaks
        }
        return cls(
            height_levels=height_levels,
            band_configs=cls.build_band_configs(height_levels)
        )

    @classmethod
    def build_band_configs(cls, height_levels: Dict[str, int]) -> Dict[TerrainType, BandConfig]:
        # Build bands from a single set of thresholds using BAND_ORDER.
        band_configs: Dict[TerrainType, BandConfig] = {}
        low = 0
        for terrain_type, threshold_key in zip(cls.BAND_ORDER, cls.BAND_THRESHOLD_KEYS):
            high = height_levels[threshold_key]
            start_color, end_color = cls.BAND_COLOR_MAP[terrain_type]
            band_configs[terrain_type] = BandConfig(low, high, start_color, end_color)
            low = high
        return band_configs

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

        # Apply band colorization in elevation order
        final = np.zeros((*Z.shape, 3), dtype=np.uint8)
        
        # Apply each band in order from lowest to highest elevation
        terrain_order = [
            TerrainType.DEEP_WATER,
            TerrainType.SHALLOW_WATER, 
            TerrainType.COASTAL,
            TerrainType.LOWLANDS,
            TerrainType.HILLS,
            TerrainType.HIGHLANDS,
            TerrainType.MOUNTAINS,
            TerrainType.PEAKS
        ]
        
        for terrain_type in terrain_order:
            if terrain_type in self.terrain_config.band_configs:
                config = self.terrain_config.band_configs[terrain_type]
                band_image = self._apply_band(Z, config)
                # Create mask for this elevation range
                mask = (Z >= config.in_min) & (Z < config.in_max)
                final[mask] = band_image[mask]

        # Parchment overlay: change weights to make the page look more or less faded.
        paper = np.full_like(final, [235, 220, 190])  # RGB parchment tone
        final = cv2.addWeighted(final, 0.85, paper, 0.15, 0)

        # Create masks for core terrain zones.
        levels = self.terrain_config.height_levels
        water_mask = Z < levels["coastal_level"]  # All water areas
        mountain_mask = Z >= levels["mountains_level"]  # Mountains and peaks
        snow_mask = Z >= levels["peaks_level"] - 10  # High peaks get snow; adjust offset for more/less snow
        cliffs = slope > 120  # Increase/decrease this to make cliff edges sharper or softer

        # Placeholder band images for APIs expecting separate band layers.
        dummy_band = final.copy()

        return (Z, dummy_band, dummy_band, dummy_band, water_mask, mountain_mask,
                snow_mask, cliffs, final)

    def _apply_band(self, depth: np.ndarray, config: BandConfig) -> np.ndarray:
        """Apply color band transformation with custom gradient."""
        # Normalize depth values to 0-1 range
        x = np.clip(depth - config.in_min, 0, config.in_max - config.in_min)
        x = x / (config.in_max - config.in_min + 1e-6)
        
        # Create RGB gradient
        start_r, start_g, start_b = config.start_color
        end_r, end_g, end_b = config.end_color
        
        r = (start_r + x * (end_r - start_r)).astype(np.uint8)
        g = (start_g + x * (end_g - start_g)).astype(np.uint8)
        b = (start_b + x * (end_b - start_b)).astype(np.uint8)
        
        # Stack into RGB image
        return np.stack([b, g, r], axis=-1)

    def generate_items(self, item_type: ItemType, canvas: np.ndarray, Z: np.ndarray,
                      region_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Generate items of specified type on canvas."""
        if item_type not in self.item_configs:
            raise ValueError(f"Unknown item type: {item_type}")

        config = self.item_configs[item_type]
        levels = self.terrain_config.height_levels

        # Create terrain suitability mask
        terrain_mask = config.terrain_mask(Z,
                                         levels["coastal_level"],
                                         levels["mountains_level"],
                                         levels["peaks_level"])

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
        ...     height_levels={"coastal_level": 100, "mountains_level": 200},
        ...     band_configs={
        ...         TerrainType.DEEP_WATER: BandConfig(0, 100, (0, 0, 180), (0, 100, 255))
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