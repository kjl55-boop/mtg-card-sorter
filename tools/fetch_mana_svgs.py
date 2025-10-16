import requests
from pathlib import Path
import time
import cairosvg
import cv2
import numpy as np

# Directories
SVG_DIR = Path("data/mana_symbols_svg")
PNG_DIR = Path("data/mana_symbols_png")
SVG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

SCRYFALL_SYMBOLS_URL = "https://api.scryfall.com/symbology"

def fetch_and_convert_symbols():
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

        filename = f"{sym_text.replace('/', '_').replace(' ', '')}"
        svg_path = SVG_DIR / f"{filename}.svg"
        png_path = PNG_DIR / f"{filename}.png"

        try:
            # Download SVG
            svg_data = requests.get(svg_url).content
            with open(svg_path, "wb") as f:
                f.write(svg_data)

            # Convert to PNG
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=64, output_height=64)

            # Optional: convert to grayscale for ORB
            img = cv2.imread(str(png_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(png_path), gray)

            print(f"Saved: {filename}.png")
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(f"Done. Saved {len(list(PNG_DIR.glob('*.png')))} symbols to {PNG_DIR}")

if __name__ == "__main__":
    fetch_and_convert_symbols()
