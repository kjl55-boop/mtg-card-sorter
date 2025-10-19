# look into https://github.com/pyenv/pyenv


# MTG Card Sorter

A modular Magic: The Gathering card sorting system using Raspberry Pi 5, Pi Camera Module 3, and Raspberry Pi Pico slave boards.

## Features
- OCR-based card recognition
- I2C communication with Pico modules
- Modular bin control (fans, servos)
- Scalable architecture

## Hardware
- Raspberry Pi 5 (master)
- Pi Camera Module 3
- Raspberry Pi Pico (slaves)
- PCA9685 servo driver (optional)
- SSD1306 OLED displays (optional)

## Setup
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/mtg-card-sorter.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`

## Roadmap
- [x] Baseline OCR
- [x] I2C communication
- [ ] Foil detection
- [ ] Bin feedback loop


## Generate / Access VM on DEV PC
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install numpy opencv-python pytesseract imagehash pillow smbus2
pip freeze > requirements.txt



🧱 Database Build & Deployment (Dev → Pi)
This section explains how to generate the Magic card database and phash index on your development machine, package it for transfer, and extract it on your Raspberry Pi.

🔧 Step 1: Build the Database (Dev Machine)
Run the builder with your desired set code (e.g. m20):

bash
python -m tools.db_builder --set m20
This will:

Fetch card data from Scryfall

Compute perceptual hashes from card images

Save metadata to data/scryfall_db/cards.db

Save phash index to data/scryfall_db/descriptors/phash_index.pkl

📦 Step 2: Package the Database
After building, compress the database and index into a zip archive:

bash
python -m tools.db_packager package
This creates:

Code
db_package.zip
├── cards.db
├── phash_index.pkl
🚚 Step 3: Transfer to Raspberry Pi
Copy db_package.zip to your Pi via SCP, USB, Git, or any preferred method.

📥 Step 4: Extract on Raspberry Pi
Once the zip is on your Pi, extract the contents:

bash
python -m tools.db_packager extract
This restores:

cards.db → for metadata and sorting

phash_index.pkl → for fast phash-based recognition


## Folder Structure
mtg-card-sorter/
├── recognizer/         # All card recognition logic
│   ├── __init__.py
│   ├── matcher.py
│   ├── phash.py
│   ├── ocr.py
│   ├── crop.py
│   ├── preprocess.py
├── hardware/           # GPIO, camera, motors, board comms
│   ├── __init__.py
│   ├── camera.py
│   ├── motor_control.py
│   ├── board_comm.py
├── pipeline/           # High-level orchestration
│   ├── __init__.py
│   ├── run_inspector.py
│   ├── capture.py
│   ├── utils.py
├── config/
│   ├── __init__.py
│   └── config.py
├── tools/
│   ├── __init__.py
│   ├── db_builder.py
│   ├── db_packager.py
│   ├── descriptors_utils.py
│   └── phash_indexer.py
├── tests/
│   ├── __init__.py
│   └── test_phash_debug_folder.py
├── data/
├── logs/
├── requirements.txt

