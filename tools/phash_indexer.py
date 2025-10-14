import sqlite3
import pickle
from pathlib import Path

def build_phash_index(db_path, output_path):
    """Build a structured phash index with raw bytes for matcher compatibility."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, phash FROM cards WHERE phash IS NOT NULL")
    index = {}

    for card_id, phash_str in cur.fetchall():
        try:
            # Convert hex string to raw bytes
            phash_bytes = bytes.fromhex(phash_str)
            index[card_id] = {
                "phash": phash_bytes,
                "phash_len": len(phash_bytes),
                "meta": {
                    "path": str(Path("data/scryfall_db/card_images") / f"{card_id}.jpg")
                }
            }
        except Exception as e:
            print(f"⚠️ Failed to convert phash for {card_id}: {e}")

    conn.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
