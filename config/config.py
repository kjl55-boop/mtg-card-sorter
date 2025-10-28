# near top: keep your existing imports and loader/profile code unchanged
from types import SimpleNamespace
from . import paths, loader  # keep your existing imports
from config.game_profiles import MTG_PROFILE, POKEMON_PROFILE

# Build CONFIG at import time from loader defaults (no parse_flags here)
CONFIG = SimpleNamespace(
    data_dir=paths.DATA_DIR,
    debug_dir=loader.DEBUG_DIR,
    results_dir=paths.RESULTS_DIR,
    logs_dir=paths.LOGS_DIR,
    descriptors_dir=paths.DESCRIPTORS_DIR,

    log_level=loader.LOG_LEVEL,
    log_format=loader.LOG_FORMAT,

    preview_size=loader.CAMERA_PREVIEW_SIZE,
    display_scale=loader.DISPLAY_SCALE,
    camera_timeout=loader.CAMERA_TIMEOUT,
    camera_warmup_sec=loader.CAMERA_WARMUP_SEC,
    autofocus_enabled=loader.AUTOFOCUS_ENABLED,

    phash_size=loader.PHASH_SIZE,
    phash_threshold=loader.PHASH_THRESHOLD,
    phash_top_k=loader.PHASH_TOP_K,
    phash_out_size=loader.PHASH_OUT_SIZE,
    phash_clahe=loader.PHASH_CLAHE,
    phash_blur_ksize=loader.PHASH_BLUR_KSIZE,
    phash_crop_margin_pct=loader.PHASH_CROP_MARGIN_PCT,
    phash_highpass=loader.PHASH_HIGHPASS,
    match_attempts=loader.MATCH_ATTEMPTS,

    save_crops=loader.SAVE_CROPS_ENABLED,

    min_area=loader.DEFAULT_MIN_AREA,
    top_pct=loader.DEFAULT_TOP_PCT,
    mid_start_pct=loader.DEFAULT_MID_START_PCT,
    mid_end_pct=loader.DEFAULT_MID_END_PCT,
    bottom_pct=loader.DEFAULT_BOTTOM_PCT,
    pad_x_pct=loader.DEFAULT_PAD_X_PCT,
    pad_y_pct=loader.DEFAULT_PAD_Y_PCT,

    normalized_size=loader.NORMALIZED_SIZE,

    game_profile=MTG_PROFILE
)


def apply_flag_overrides(args):
    """
    Apply non-None parsed CLI args into CONFIG in-place.
    Call this once early in the process after parse_flags().
    """
    if not args:
        return

    # Logging
    if getattr(args, "log_level", None) is not None:
        CONFIG.log_level = args.log_level

    # Camera
    if getattr(args, "preview_width", None) and getattr(args, "preview_height", None):
        CONFIG.preview_size = (args.preview_width, args.preview_height)
    if getattr(args, "display_scale", None) is not None:
        CONFIG.display_scale = args.display_scale

    # Matching
    if getattr(args, "phash_size", None) is not None:
        CONFIG.phash_size = args.phash_size
    if getattr(args, "phash_threshold", None) is not None:
        CONFIG.phash_threshold = args.phash_threshold
    if getattr(args, "phash_top_k", None) is not None:
        CONFIG.phash_top_k = args.phash_top_k
    if getattr(args, "match_attempts", None) is not None:
        CONFIG.match_attempts = args.match_attempts

    # Output
    if getattr(args, "debug_dir", None) is not None:
        CONFIG.debug_dir = args.debug_dir
    if getattr(args, "save_crops", None) is not None:
        CONFIG.save_crops = bool(args.save_crops)

    # add any additional fields you rely on similarly...
