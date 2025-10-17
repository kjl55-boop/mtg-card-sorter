import sqlite3
import pickle
from pathlib import Path

def build_phash_index(db_path, output_path):
    """Build a structured phash index using hex strings."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, phash, phash_title, phash_mana, phash_text, name FROM cards WHERE phash IS NOT NULL")
    index = {}

    for card_id, phash_str, phash_title, phash_mana, phash_text, name in cur.fetchall():
        if phash_str:
            index[card_id] = {
                "phash": phash_str,
                "phash_title": phash_title,
                "phash_mana": phash_mana,
                "phash_text": phash_text,
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
