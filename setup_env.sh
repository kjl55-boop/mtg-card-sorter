#!/bin/bash

# Activate venv
source venv/bin/activate

# Step 1: Install system packages
echo "Installing system packages via apt..."
sudo apt update

# Read apt-packages.txt and install each one
while read -r package; do
    if dpkg -s "$package" &> /dev/null; then
        echo "$package already installed"
    else
        echo "Installing $package..."
        sudo apt install -y "$package"
    fi
done < apt-packages.txt

# Step 2: Install pip packages
echo "Installing pip packages into venv..."
pip install -r requirements.txt

echo "✅ Environment setup complete."
