import sqlite3
import pickle
from pathlib import Path

def build_phash_index(db_path, output_path):
    """Build a dictionary of {card_id: phash} from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, phash FROM cards WHERE phash IS NOT NULL")
    index = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
