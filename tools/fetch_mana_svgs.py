import requests
from pathlib import Path
import time

# Output directory
OUT_DIR = Path("data/mana_symbols")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scryfall API endpoint
SCRYFALL_SYMBOLS_URL = "https://api.scryfall.com/symbology"

def fetch_mana_svgs():
    print("Fetching mana symbols from Scryfall...")
    try:
        response = requests.get(SCRYFALL_SYMBOLS_URL)
        response.raise_for_status()
        symbols = response.json().get("data", [])
    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return

    for symbol in symbols:
        sym_text = symbol.get("symbol", "").strip("{}")
        svg_url = symbol.get("svg_uri")
        if not svg_url or not sym_text:
            continue

        # Normalize filename
        filename = f"{sym_text.replace('/', '_').replace(' ', '')}.svg"
        out_path = OUT_DIR / filename

        try:
            svg_data = requests.get(svg_url).content
            with open(out_path, "wb") as f:
                f.write(svg_data)
            print(f"Saved: {filename}")
            time.sleep(0.1)  # Be polite to Scryfall
        except Exception as e:
            print(f"Failed to save {filename}: {e}")

    print(f"Done. Saved {len(list(OUT_DIR.glob('*.svg')))} symbols to {OUT_DIR}")

if __name__ == "__main__":
    fetch_mana_svgs()
