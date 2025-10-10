#!/usr/bin/env python3
"""
Batch PHASH test tool

Usage:
  python tools/batch_phash_test.py             # uses defaults: input_dir=data/debug, db=data/scryfall_db/cards.db
  python tools/batch_phash_test.py --input dir --db path --out results.csv --max-cand 5 --thr 8

Output:
  - CSV with rows: image_path,query_phash,match_type,match_id,match_name,match_phash,hamming
    match_type: exact | prefix | top_n
"""

import argparse
from pathlib import Path
from PIL import Image
import imagehash
import sqlite3
import csv
import sys

def hamming(a_hex, b_hex):
    ai = int(str(a_hex), 16)
    bi = int(str(b_hex), 16)
    return bin(ai ^ bi).count("1")

def load_db_phashes(db_path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, name, phash FROM cards WHERE phash IS NOT NULL AND phash != ''")
    rows = cur.fetchall()
    conn.close()
    return rows

def find_best_candidates(query_ph, rows, top_n=5):
    scores = []
    for cid, name, ph in rows:
        try:
            d = hamming(query_ph, ph)
            scores.append((d, cid, name, ph))
        except Exception:
            continue
    scores.sort(key=lambda x: x[0])
    return scores[:top_n]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="data/debug", help="Input directory with images")
    p.add_argument("--db", "-d", default="data/scryfall_db/cards.db", help="SQLite cards DB")
    p.add_argument("--out", "-o", default="results/batch_phash_results.csv", help="Output CSV file")
    p.add_argument("--max-cand", "-n", type=int, default=5, help="Top N candidates to record")
    p.add_argument("--threshold", "-t", type=int, default=8, help="Hamming threshold for a 'top match'")
    args = p.parse_args()

    input_dir = Path(args.input)
    db_path = Path(args.db)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(2)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(2)

    print("Loading DB phashes...")
    rows = load_db_phashes(db_path)
    if not rows:
        print("No phashes found in DB. Populate cards.db phash column first.", file=sys.stderr)
        sys.exit(3)
    print(f"Loaded {len(rows)} DB phash rows")

    images = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg")])
    if not images:
        print("No images found in input directory.", file=sys.stderr)
        sys.exit(4)
    print(f"Found {len(images)} images to test")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path","query_phash","match_type","match_id","match_name","match_phash","hamming"])
        for img_path in images:
            try:
                ph = str(imagehash.phash(Image.open(img_path).convert("RGB")))
            except Exception as e:
                print(f"Failed to open/hash {img_path}: {e}", file=sys.stderr)
                continue

            # exact / prefix checks via DB rows
            exact = [r for r in rows if r[2] == ph]
            prefix = [r for r in rows if isinstance(r[2], str) and r[2].startswith(ph[:8])]

            if exact:
                for cid, name, ph_db in exact:
                    writer.writerow([str(img_path), ph, "exact", cid, name, ph_db, 0])
                print(f"{img_path.name}: exact match found ({len(exact)})")
                continue
            if prefix:
                for cid, name, ph_db in prefix:
                    # compute hamming for reporting
                    d = hamming(ph, ph_db)
                    writer.writerow([str(img_path), ph, "prefix", cid, name, ph_db, d])
                print(f"{img_path.name}: prefix match found ({len(prefix)})")
                continue

            # Hamming search
            best = find_best_candidates(ph, rows, top_n=args.max_cand)
            recorded = 0
            for d, cid, name, ph_db in best:
                writer.writerow([str(img_path), ph, "top_n", cid, name, ph_db, d])
                recorded += 1
            if best and best[0][0] <= args.threshold:
                print(f"{img_path.name}: candidate match within threshold ({best[0][0]}) -> {best[0][1]} {best[0][2]}")
            else:
                print(f"{img_path.name}: No close match; best hamming {best[0][0] if best else 'N/A'}")
    print("Results written to", out_path)

if __name__ == "__main__":
    main()
