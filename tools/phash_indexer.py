import sqlite3
import pickle
from pathlib import Path

def build_phash_index(db_path, output_path):
    """Build a structured phash index using hex strings."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, phash, name FROM cards WHERE phash IS NOT NULL")
    index = {}

    for card_id, phash_str, name in cur.fetchall():
        if phash_str:
            index[card_id] = {
                "phash": phash_str,
                "phash_len": len(phash_str),
                "meta": {
                    "path": str(Path("data/scryfall_db/card_images") / f"{card_id}.jpg"),
                    "name": name
                }
            }

    conn.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
