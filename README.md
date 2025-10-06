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