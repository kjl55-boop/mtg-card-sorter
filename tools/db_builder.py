# orchestrates full build via ScryfallDBBuilder class

import json, sqlite3, logging, requests, time
from pathlib import Path
from tools.descriptor_utils import phash_for_image_path, save_phash_descriptor
from tools.phash_indexer import build_phash_index

from PIL import Image
from io import BytesIO
import requests
import imagehash

class ScryfallDBBuilder:
    def __init__(self, set_code="m20", root_dir=Path(__file__).resolve().parents[1]):
        self.set_code = set_code
        self.root = Path(root_dir)
        self.data_dir = self.root / "data" / "scryfall_db"
        self.raw_dir = self.data_dir / "raw_scryfall"
        self.desc_dir = self.data_dir / "descriptors"
        self.db_path = self.data_dir / "cards.db"
        self.json_path = self.data_dir / f"scryfall_{set_code}.json"
        self.index_path = self.desc_dir / "phash_index.pkl"
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"builder_{self.set_code}")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    def fetch_cards(self):
        url = "https://api.scryfall.com/cards/search"
        params = {"q": f"set:{self.set_code}", "unique": "prints", "order": "set"}
        cards = []
        while url:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            cards.extend(data["data"])
            url = data.get("next_page")
            params = None
            time.sleep(0.1)
        with open(self.json_path, "w") as f:
            json.dump(cards, f, indent=2)
        self.logger.info("Fetched %d cards", len(cards))
        return cards

    def process_card(self, card):
        card_id = card.get("id") or card.get("oracle_id") or card.get("name", "unknown")
        card_id = card_id.replace(" ", "_")
        img_url = card.get("image_url") or card.get("image_uris", {}).get("normal")

        if img_url and img_url.startswith("http"):
            try:
                response = requests.get(img_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                phash = str(imagehash.phash(image))
                save_phash_descriptor(card_id, phash, self.desc_dir)
                return card_id, phash
            except Exception as e:
                self.logger.warning("Failed to process image for %s: %s", card_id, e)

        return card_id, None

    def build_db(self, cards):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY,
                name TEXT,
                set_code TEXT,
                collector_number TEXT,
                image_url TEXT,
                phash TEXT,
                colors TEXT,
                type_line TEXT,
                mana_cost TEXT,
                rarity TEXT,
                usd_price TEXT
            )
        """)
        for card in cards:
            card_id, phash = self.process_card(card)
            #colors = ",".join(card.get("colors", []))
            #type_line = card.get("type_line")
            #mana_cost = card.get("mana_cost")
            #rarity = card.get("rarity")
            #usd_price = card.get("prices", {}).get("usd")
            cur.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                card_id,
                card.get("name"),
                card.get("set_code"),
                card.get("collector_number"),
                card.get("image_url"),
                phash,
                ",".join(card.get("colors", [])),
                card.get("type_line"),
                card.get("mana_cost"),
                card.get("rarity"),
                card.get("prices", {}).get("usd")
            ))
        conn.commit()
        conn.close()

    def build_index(self):
        build_phash_index(self.db_path, self.index_path)

    def run(self):
        cards = self.fetch_cards()
        self.build_db(cards)
        self.build_index()
        self.logger.info("Build complete for set %s", self.set_code)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Magic card database and phash index.")
    parser.add_argument("--set", type=str, default="m20", help="Set code to build (e.g. m20, khm, eld)")
    args = parser.parse_args()

    builder = ScryfallDBBuilder(set_code=args.set)
    builder.run()
    print(f"✅ Build complete for set {args.set}")


