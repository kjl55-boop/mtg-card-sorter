import pickle

index_path = "data/scryfall_db/descriptors/phash_index.pkl"

with open(index_path, "rb") as f:
    index = pickle.load(f)

print(f"Total entries: {len(index)}\n")

for card_id, rec in list(index.items())[:5]:
    print(f"{card_id}: len(phash) = {len(rec['phash'])}")

import pickle

with open("data/scryfall_db/descriptors/phash_index.pkl", "rb") as f:
    index = pickle.load(f)

rec = index.get("13becea1-e745-4c96-bfc2-6a277fb60ee1")
print(rec["meta"].get("name"))


# Show first 5 entries
for i, (card_id, value) in enumerate(index.items()):
    print(f"{i+1}. Card ID: {card_id}")
    if isinstance(value, dict):
        print(f"   Phash: {value.get('phash')}")
        print(f"   Meta: {value.get('meta')}")
    else:
        print(f"   Phash: {value}")
    if i >= 4:
        break
