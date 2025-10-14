import pickle

index_path = "data/scryfall_db/descriptors/phash_index.pkl"

with open(index_path, "rb") as f:
    index = pickle.load(f)

print(f"Total entries: {len(index)}\n")

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
