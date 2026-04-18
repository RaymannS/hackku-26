import numpy as np
import cv2
import random
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Import render functions
try:
    from .render import (
        draw_tree, draw_cactus, draw_house, draw_label, draw_castle,
        apply_desert_terrain, draw_contour_lines,
        apply_hillshade, apply_paper_texture, apply_vignette
    )
except ImportError:
    # For testing purposes
    def draw_tree(*args): pass
    def draw_cactus(*args): pass  
    def draw_house(*args): pass
    #def draw_label(*args): pass
    def apply_desert_terrain(*args): pass
    def draw_contour_lines(*args): return args[0]
    def apply_hillshade(*args): return args[0]
    def apply_paper_texture(*args): return args[0]
    def apply_vignette(*args): return args[0]

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
    CASTLE = "castle"

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
        TerrainType.DEEP_WATER: ((0, 150, 255), (0, 150, 255)),          # Bright deep ocean blue
        TerrainType.SHALLOW_WATER: ((0, 150, 255), (0, 255, 255)),     # Brighter coastal blue
        TerrainType.COASTAL: ((240, 210, 160), (215, 160, 100)),          # Warm glowing sand
        TerrainType.LOWLANDS: ((145, 195, 130), (115, 160, 100)),         # Brighter lowland green
        TerrainType.HILLS: ((115, 165, 110), (90, 135, 75)),              # Lush hill green
        TerrainType.HIGHLANDS: ((105, 135, 115), (85, 115, 95)),          # Lighter highland green
        TerrainType.MOUNTAINS: ((120, 120, 120), (170, 170, 170)),        # Slightly darker stone gray
        TerrainType.PEAKS: ((215, 215, 225), (255, 255, 255))             # Bright snowy peaks
    }

    @classmethod
    def default(cls) -> 'TerrainConfig':
        # Change elevation thresholds here. Only this block is required to adjust band heights.
        height_levels = {
            "deep_water_level": 45,      # Max elevation of deep ocean
            "shallow_water_level": 78,   # Max elevation of shallow ocean
            "coastal_level": 88,         # Max elevation of beaches/coastal plains
            "lowlands_level": 115,       # Max elevation of lowland plains
            "hills_level": 155,          # Max elevation of rolling hills
            "highlands_level": 190,      # Max elevation of highlands
            "mountains_level": 215,      # Max elevation of mountain slopes
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
    cluster_radius: int = 80  # ← add this

class SceneGenerator:
    def __init__(self, terrain_config: Optional[TerrainConfig] = None, style_enabled: bool = True):
        self.terrain_config = terrain_config or TerrainConfig.default()
        self.style_enabled = style_enabled
        self.item_configs = self._setup_item_configs()

    def set_style_enabled(self, enabled: bool):
        self.style_enabled = enabled

    def toggle_style(self) -> bool:
        self.style_enabled = not self.style_enabled
        return self.style_enabled

    def _setup_item_configs(self) -> Dict[ItemType, ItemConfig]:
        """Setup default item configurations."""
        return {
            ItemType.FOREST: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea) & (Z < mountain),
                count_range=(40, 60),
                min_distance=1,
                draw_function=lambda canvas, x, y: draw_tree(canvas, x, y),
                cluster_radius=80
            ),
            ItemType.DESERT: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea) & (Z < sea + 50),
                count_range=(30, 45),
                min_distance=25,
                draw_function=lambda canvas, x, y: draw_cactus(canvas, x, y),
                cluster_radius=80
            ),
            ItemType.TOWN: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < mountain - 20),
                count_range=(3, 8),
                min_distance=15,
                draw_function=lambda canvas, x, y: draw_house(canvas, x, y),
                cluster_radius=40
            ),
            ItemType.VILLAGE: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < mountain - 20),
                count_range=(2, 5),
                min_distance=12,
                draw_function=lambda canvas, x, y: draw_house(canvas, x, y),
                cluster_radius=40
            ),
            ItemType.CITY: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < mountain - 20),
                count_range=(8, 15),
                min_distance=18,
                draw_function=lambda canvas, x, y: draw_house(canvas, x, y),
                cluster_radius=60
            ),
            ItemType.CASTLE: ItemConfig(
                terrain_mask=lambda Z, sea, mountain, snow: (Z >= sea + 5) & (Z < snow),
                count_range=(1,1),
                min_distance=0,
                draw_function=lambda canvas, x, y: draw_castle(canvas, x, y),
                cluster_radius=100
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

        if self.style_enabled:
            # Smooth transitions between elevation bands.
            blurred = cv2.GaussianBlur(final, (5, 5), 0)
            final = cv2.addWeighted(final, 0.94, blurred, 0.06, 0)

            # Add subtle hillshade to make relief feel more three-dimensional.
            final = apply_hillshade(final, Z)

            # Parchment overlay: use a lighter tint so the terrain stays bright.
            paper = np.full_like(final, [240, 230, 205])  # Brighter parchment tone
            final = cv2.addWeighted(final, 0.92, paper, 0.08, 0)

            # Add topographic contour lines for a map-like look.
            final = draw_contour_lines(final, Z, interval=20, color=(70, 45, 30), thickness=1, alpha=0.14)

            # Add subtle texture and vignette to improve visual richness.
            final = apply_paper_texture(final, intensity=0.03)
            final = apply_vignette(final, strength=0.08)
        else:
            paper = np.full_like(final, [235, 220, 190])
            final = cv2.addWeighted(final, 0.92, paper, 0.08, 0)

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
        positions = self._scatter_placer(terrain_mask, count, config.min_distance, config.cluster_radius)


        # Draw items
        for x, y in positions:
            config.draw_function(canvas, x, y)

        return positions

    def _scatter_placer(self, mask: np.ndarray, n: int, min_dist: int, radius: int = 80) -> List[Tuple[int, int]]:
        coords = np.argwhere(mask).tolist()
        if not coords:
            return []

        h, w = mask.shape

        # only consider center points that are at least radius away from edges
        interior = [(y, x) for y, x in coords 
                    if x >= radius and x < w - radius and y >= radius and y < h - radius]
        
        if not interior:
            interior = coords  # fallback if map is too small

        cy, cx = random.choice(interior)

        nearby = [(y, x) for y, x in coords if np.sqrt((x - cx)**2 + (y - cy)**2) <= radius]
        if not nearby:
            nearby = coords

        random.shuffle(nearby)
        placed = []

        for y, x in nearby:
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