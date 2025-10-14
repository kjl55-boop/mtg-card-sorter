import sqlite3
import pickle
from pathlib import Path

def build_phash_index(db_path, output_path):
    """Build a structured phash index compatible with matcher expectations."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, phash FROM cards WHERE phash IS NOT NULL")
    index = {}

    for card_id, phash_str in cur.fetchall():
        if phash_str:
            phash_bytes = bytes.fromhex(phash_str)  # convert hex string to bytes
            index[card_id] = {
                "phash": phash_bytes,
                "phash_len": len(phash_bytes),
                "meta": {
                    "path": str(Path("data/scryfall_db/card_images") / f"{card_id}.jpg")
                }
            }

    conn.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
