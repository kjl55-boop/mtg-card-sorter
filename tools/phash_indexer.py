import sqlite3
import pickle
from pathlib import Path

def build_phash_index(db_path, output_path):
    """Build a structured phash index from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, phash FROM cards WHERE phash IS NOT NULL")
    index = {}

    for card_id, phash in cur.fetchall():
        if phash:
            index[card_id] = {
                "phash": phash,
                "phash_len": len(phash),
                "meta": {
                    # Update this path if your card images are stored elsewhere
                    "path": str(Path("data/scryfall_db/card_images") / f"{card_id}.jpg")
                }
            }

    conn.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
