"""
Unified runtime config combining paths, static defaults, and CLI flags.
Use CONFIG to access resolved values across the pipeline.
"""

from types import SimpleNamespace
from . import paths, loader, flags
from config.game_profiles import MTG_PROFILE, POKEMON_PROFILE
from typing import Optional

# Lazy cached storage
_ARGS = None
_CONFIG = None

def get_runtime_args(argv: Optional[list] = None):
    """
    Lazily parse CLI flags on first request and cache the result.
    Call this only from guarded entrypoints or when you actually need runtime args.
    """
    global _ARGS
    if _ARGS is None:
        # import inside function to avoid parse at module import time
        from config.flags import parse_flags
        _ARGS = parse_flags(argv)
    return _ARGS

def _build_config_from_args(args):
    """
    Build and return a SimpleNamespace CONFIG using the provided args (Namespace).
    This function mirrors your previous top-level CONFIG construction but does not run
    at import time.
    """
    return SimpleNamespace(
        # Paths
        data_dir=paths.DATA_DIR,
        debug_dir=(args.debug_dir or loader.DEBUG_DIR),
        results_dir=paths.RESULTS_DIR,
        logs_dir=paths.LOGS_DIR,
        descriptors_dir=paths.DESCRIPTORS_DIR,

        # Logging
        log_level=(args.log_level or loader.LOG_LEVEL),
        log_format=loader.LOG_FORMAT,

        # Camera
        preview_size=( (args.preview_width, args.preview_height)
                       if getattr(args, "preview_width", None) and getattr(args, "preview_height", None)
                       else loader.CAMERA_PREVIEW_SIZE),
        display_scale=(args.display_scale or loader.DISPLAY_SCALE),
        camera_timeout=loader.CAMERA_TIMEOUT,
        camera_warmup_sec=loader.CAMERA_WARMUP_SEC,
        autofocus_enabled=loader.AUTOFOCUS_ENABLED,

        # Matching
        phash_size=(args.phash_size or loader.PHASH_SIZE),
        phash_threshold=(args.phash_threshold or loader.PHASH_THRESHOLD),
        phash_top_k=(args.phash_top_k or loader.PHASH_TOP_K),
        phash_out_size = loader.PHASH_OUT_SIZE,
        phash_clahe = loader.PHASH_CLAHE,
        phash_blur_ksize = loader.PHASH_BLUR_KSIZE,
        phash_crop_margin_pct = loader.PHASH_CROP_MARGIN_PCT,
        phash_highpass = loader.PHASH_HIGHPASS,

        match_attempts=(args.match_attempts or loader.MATCH_ATTEMPTS),

        # Output
        save_crops=(args.save_crops or loader.SAVE_CROPS_ENABLED),

        # Crop defaults
        min_area=loader.DEFAULT_MIN_AREA,
        top_pct=loader.DEFAULT_TOP_PCT,
        mid_start_pct=loader.DEFAULT_MID_START_PCT,
        mid_end_pct=loader.DEFAULT_MID_END_PCT,
        bottom_pct=loader.DEFAULT_BOTTOM_PCT,
        pad_x_pct=loader.DEFAULT_PAD_X_PCT,
        pad_y_pct=loader.DEFAULT_PAD_Y_PCT,

        # Normalization
        normalized_size=loader.NORMALIZED_SIZE,

        # Card Profile
        game_profile=MTG_PROFILE
    )

def get_config(argv: Optional[list] = None):
    """
    Return a cached CONFIG (SimpleNamespace). Builds it on first access using
    get_runtime_args(argv) so flags are applied lazily and only when needed.
    """
    global _CONFIG
    if _CONFIG is None:
        args = get_runtime_args(argv)
        _CONFIG = _build_config_from_args(args)
    return _CONFIG
