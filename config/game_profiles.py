from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class CardGameProfile:
    name: str
    db_path: Path
    index_path: Path
    crop_regions: Dict[str, Tuple[float, float]]  # e.g. {"title": (0.05, 0.15)}
    phash_size: int = 8
    phash_threshold: int = 10
    preprocess_config: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Game Profiles
# ─────────────────────────────────────────────────────────────

#CropRegion(name, x_start_pct, x_end_pct, y_start_pct, y_end_pct)

MTG_PROFILE = CardGameProfile(
    name="mtg",
    db_path=Path("data/scryfall_db/cards.db"),
    index_path=Path("data/scryfall_db/descriptors/phash_index.pkl"),
    crop_regions={
        "title": (0.05, 0.95, 0.02, 0.12),
        "mana": (0.75, 0.98, 0.02, 0.12),
        "text": (0.05, 0.95, 0.55, 0.75),
        "bottom": (0.05, 0.95, 0.78, 0.98)
    },
    preprocess_config={
        "clahe": True,
        "blur_ksize": (3, 3),
        "crop_margin_pct": 0.02
    }
)

POKEMON_PROFILE = CardGameProfile(
    name="pokemon",
    db_path=Path("data/pokemon_db/cards.db"),
    index_path=Path("data/pokemon_db/descriptors/phash_index.pkl"),
    crop_regions={
        "title": (0.08, 0.18),
        "hp": (0.18, 0.28),
        "text": (0.30, 0.90)
    },
    preprocess_config={
        "clahe": False,
        "blur_ksize": (5, 5),
        "crop_margin_pct": 0.03
    }
)