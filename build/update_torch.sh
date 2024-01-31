#!/bin/bash

# Uninstall previous PyTorch version
echo "Uninstalling previous PyTorch version..."
pip uninstall torch torchvision torchaudio

# Install the nightly version
echo "Installing the nightly version of PyTorch..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# If you want to use pip3 instead of pip, you can replace the above commands with the following:
# pip3 uninstall torch torchvision torchaudio
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

echo "PyTorch nightly version installed successfully!"