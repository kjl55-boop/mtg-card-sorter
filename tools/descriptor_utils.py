from PIL import Image
import imagehash
from pathlib import Path
import pickle

def phash_for_image_path(img_path):
    """Compute perceptual hash string for an image file path."""
    im = Image.open(img_path).convert("RGB")
    return str(imagehash.phash(im))

def save_phash_descriptor(card_id, phash, output_dir):
    """Save phash descriptor as a .pkl file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{card_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(phash, f)
