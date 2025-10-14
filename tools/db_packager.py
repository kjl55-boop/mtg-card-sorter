import zipfile
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "scryfall_db"
DB_PATH = DATA_DIR / "cards.db"
INDEX_PATH = DATA_DIR / "descriptors" / "phash_index.pkl"
ARCHIVE_PATH = ROOT / "db_package.zip"

logger = logging.getLogger("db_packager")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def package_db():
    """Compress cards.db and phash_index.pkl into a zip archive."""
    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(DB_PATH, arcname="cards.db")
        zipf.write(INDEX_PATH, arcname="phash_index.pkl")
    logger.info("Packaged database into %s", ARCHIVE_PATH)

def extract_db(target_dir=DATA_DIR):
    """Extract the zip archive into the target directory."""
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zipf:
        zipf.extract("cards.db", path=target_dir)
        zipf.extract("phash_index.pkl", path=target_dir / "descriptors")
    logger.info("Extracted database files to %s", target_dir)

def verify_package():
    """Check that both files exist and are readable."""
    if not DB_PATH.exists():
        logger.warning("Missing cards.db")
    if not INDEX_PATH.exists():
        logger.warning("Missing phash_index.pkl")
    if DB_PATH.exists() and INDEX_PATH.exists():
        logger.info("Both files are present and ready for packaging.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m tools.db_packager [package|extract|verify]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "package":
        verify_package()
        package_db()
    elif cmd == "extract":
        extract_db()
    elif cmd == "verify":
        verify_package()
    else:
        print("Unknown command:", cmd)
