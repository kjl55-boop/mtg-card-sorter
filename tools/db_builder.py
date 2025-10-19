import json, sqlite3, logging, requests, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.phash_indexer import build_phash_index
from PIL import Image
from io import BytesIO
import imagehash
from recognizer.card_slicer import CardSlicer
import cv2
import numpy as np
from config.config import CONFIG


class ScryfallDBBuilder:
    def __init__(self, set_code="m20", root_dir=Path(__file__).resolve().parents[1]):
        self.set_code = set_code
        self.root = Path(root_dir)
        self.data_dir = self.root / "data" / "scryfall_db"
        self.raw_dir = self.data_dir / "raw_scryfall"
        self.desc_dir = self.data_dir / "descriptors"
        self.db_path = self.data_dir / "cards.db"
        self.json_path = self.raw_dir / f"{set_code}.json"
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
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists():
            self.logger.info("Using cached Scryfall JSON: %s", self.json_path)
            with open(self.json_path, "r") as f:
                return json.load(f)

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
        self.logger.info("Fetched %d cards and saved to %s", len(cards), self.json_path)
        return cards

    def process_card(self, card):
        card_id = card.get("id") or card.get("oracle_id") or card.get("name", "unknown")
        card_id = card_id.replace(" ", "_")
        img_url = card.get("image_url") or card.get("image_uris", {}).get("normal")

        if img_url and img_url.startswith("http"):
            try:
                response = requests.get(img_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                full_phash = str(imagehash.phash(image, hash_size=16))

                # Convert to OpenCV format
                cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Slice regions
                slicer = CardSlicer()
                region_hashes = {}
                for name, crop in slicer.crop_all(cv_img).items():
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    region_hashes[name] = str(imagehash.phash(pil_crop, hash_size=CONFIG.phash_size))

                return {
                    "id": card_id,
                    "name": card.get("name"),
                    "set_code": card.get("set"),
                    "collector_number": card.get("collector_number"),
                    "image_url": img_url,
                    "phash": full_phash,
                    "phash_title": region_hashes.get("title"),
                    "phash_mana": region_hashes.get("mana"),
                    "phash_text": region_hashes.get("text"),
                    "colors": ",".join(card.get("colors", [])),
                    "type_line": card.get("type_line"),
                    "mana_cost": card.get("mana_cost"),
                    "rarity": card.get("rarity"),
                    "usd_price": card.get("prices", {}).get("usd")
                }
            except Exception as e:
                self.logger.warning("Failed to process image for %s: %s", card_id, e)
        return None


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
                phash_title TEXT,
                phash_mana TEXT,
                phash_text TEXT,
                colors TEXT,
                type_line TEXT,
                mana_cost TEXT,
                rarity TEXT,
                usd_price TEXT
            )
        """)

        # Load existing IDs to skip duplicates
        cur.execute("SELECT id FROM cards")
        existing_ids = set(row[0] for row in cur.fetchall())

        rows = []

        def process_card_if_new(card):
            card_id = card.get("id") or card.get("oracle_id") or card.get("name", "unknown")
            card_id = card_id.replace(" ", "_")
            if card_id in existing_ids:
                return None
            return self.process_card(card)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_card_if_new, card) for card in cards]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and result["phash"]:
                        rows.append((
                            result["id"],
                            result["name"],
                            result["set_code"],
                            result["collector_number"],
                            result["image_url"],
                            result["phash"],
                            result["phash_title"],
                            result["phash_mana"],
                            result["phash_text"],
                            result["colors"],
                            result["type_line"],
                            result["mana_cost"],
                            result["rarity"],
                            result["usd_price"]
                        ))
                except Exception as e:
                    self.logger.warning("Error during card processing: %s", e)

        cur.executemany("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        self.logger.info("Inserted %d new cards into database", len(rows))

    def build_index(self):
        if self.index_path.exists():
            self.logger.info("Index already exists: %s", self.index_path)
            return
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
